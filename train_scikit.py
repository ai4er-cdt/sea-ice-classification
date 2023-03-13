"""
AI4ER GTC - Sea Ice Classification
Script for feeding training and validation data into 
scikit-learn classifiers saving the model output to wandb
"""
import os
import wandb
import numpy as np
import multiprocessing as mp
from pathlib import Path
from timeit import default_timer
from joblib import dump
from constants import new_classes, model_parameters, chart_sar_pairs
from util_scikit import load_chart, load_sar, crop_image
from argparse import ArgumentParser, BooleanOptionalAction


if __name__ == '__main__':
    
    parser = ArgumentParser(description="Sea Ice Random Forest Train")
    parser.add_argument("--name", default="default", type=str, help="Name of wandb run")
    parser.add_argument("--sample", action=BooleanOptionalAction, help="Run a sample of the dataset")
    parser.add_argument("--n_sample", default=100, type=int, help="Number of tiles to use in the sample")
    parser.add_argument("--load_parallel", action=BooleanOptionalAction, help='Whether to read tiles in parallel')
    parser.add_argument("--classification_type", default="binary", type=str,
                        choices=["binary", "ternary", "multiclass"], help="Type of classification task")
    parser.add_argument("--sar_band3", default="angle", type=str, choices=["angle", "ratio"],
                        help="Whether to use incidence angle or HH/HV ratio in third band")
    parser.add_argument("--sar_folder", default='sar', type=str, help="SAR input folder name")
    parser.add_argument("--chart_folder", default='chart', type=str, help="Ice Chart input folder name")
    parser.add_argument("--model", default='RandomForest', type=str, 
                        choices=['RandomForest', 'DecisionTree', 'KNeighbors', 'SGD', 'MLP', 'SVC'], help="Classification model to use")
    parser.add_argument("--grid_search", action=BooleanOptionalAction, help='Wether to perform grid search cross-validation')
    parser.add_argument("--cv_fold", default=5, type=int, help="Number of folds for cross-validation")
    parser.add_argument("--n_cores", default=-1, type=int, help="Number of jobs to run in parallel")
    parser.add_argument("--data_type", default='tile', type=str, choices=['tile', 'original'], help='Run the classifier on the tiles or the original images')
    parser.add_argument("--flip_vertically", action=BooleanOptionalAction,
                        help="Whether to flip an ice chart vertically to match the SAR coordinates")
    parser.add_argument("--impute", action=BooleanOptionalAction,
                        help="Whether to impute missing values in SAR and Ice charts")
    parser.add_argument("--seed", default=0, type=int, help="Numpy random seed")
    args = parser.parse_args()
    
    t_start = default_timer()
    
    class_categories = new_classes[args.classification_type]
    n_classes = len(class_categories)
    sar_band3 = args.sar_band3
    is_binary = True if args.classification_type == 'binary' else False
    seed = np.random.seed(args.seed)
    
    def load_sar_wrapper(file_path: str):
        return load_sar(file_path, sar_band3)
        
    def load_chart_wrapper(file_path: str):
        return load_chart(file_path, class_categories)
    
    def load_chart_wrapper_vertical(file_path: str):
        return load_chart(file_path, class_categories, flip_vertically=args.flip_vertically)
    
    # standard input dirs
    
    if args.data_type == 'tile':
        input_folder = Path(open("tile.config").read().strip())
        sar_folder = f"{input_folder}/{args.sar_folder}"
        chart_folder = f"{input_folder}/{args.chart_folder}"
        
        sar_filenames = os.listdir(sar_folder)
        chart_filenames = os.listdir(chart_folder)
        sar_filenames = [os.path.join(sar_folder, x) for x in sar_filenames]
        chart_filenames = [os.path.join(chart_folder, x) for x in chart_filenames]
        
    elif args.data_type == 'original':
        input_folder = Path(open("ftp.config").read().strip())
        sar_folder = f"{input_folder}/dual_band_images"
        chart_folder = f"{input_folder}/rasterised_shapefiles"
        chart_ext = "tiff"
        sar_ext = "tif"
        
        sar_filenames = [os.path.join(sar_folder, f'{sar}.{sar_ext}') for (_, sar, _) in chart_sar_pairs]
        chart_filenames = [os.path.join(chart_folder, f'{chart}.{chart_ext}') for (chart, _, _) in chart_sar_pairs]
    
    if args.sample:
        assert args.n_sample <= len(sar_filenames)
        sample_n = np.random.randint(len(sar_filenames), size=(args.n_sample))
        sar_filenames = [sar_filenames[i] for i in sample_n]
        chart_filenames = [chart_filenames[i] for i in sample_n]
        
    if args.load_parallel:
        print('Loading tiles in parallel')
        cores = mp.cpu_count() if args.n_cores == -1 else args.n_cores
        mp_pool = mp.Pool(cores)
        
        train_x_lst = mp_pool.map(load_sar_wrapper, sar_filenames)
        if args.data_type == 'original' and args.flip_vertically:
            train_y_lst = mp_pool.map(load_chart_wrapper_vertical, chart_filenames)
        else:
            train_y_lst = mp_pool.map(load_chart_wrapper, chart_filenames)
        
        mp_pool.close()
    else:
        print('Loading tiles')
        train_x_lst = [load_sar(sar, sar_band3=sar_band3) for sar in sar_filenames]
        train_y_lst = [load_chart(chart, class_categories, flip_vertically=args.flip_vertically) for chart in chart_filenames]
        
    if args.data_type == 'original':
        height_min = 100000000
        width_min = 100000000
        height_max = 0
        width_max = 0
        for chart in train_y_lst:
            shape = chart.shape
            if shape[1] < height_min:
                height_min = shape[1]
            if shape[2] < width_min:
                width_min = shape[2]
        
        dim_min = min([height_min, width_min])
                
        train_x_lst = [crop_image(sar, height_min, width_min) for sar in train_x_lst]
        train_y_lst = [crop_image(chart, height_min, width_min) for chart in train_y_lst]

    train_x = np.stack(train_x_lst)
    train_y = np.stack(train_y_lst)

    # Reorder dimensions
    X_train_data = np.moveaxis(train_x, 1, -1).reshape(-1, 3)
    Y_train_data = np.moveaxis(train_y, 1, -1).reshape(-1, 1)

    if args.impute:
        from sklearn.impute import KNNImputer
        x_imputer = KNNImputer(n_neighbors=8)
        y_imputer = KNNImputer(n_neighbors=8)
        
        X_train_data = x_imputer.fit_transform(X_train_data)
        Y_train_data = y_imputer.fit_transform(Y_train_data)        
            
    # Models
    print(f'Training {args.model}')
    if args.model == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_jobs=args.n_cores, random_state=seed)
    elif args.model == 'DecisionTree':
        from sklearn.tree import  DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=seed)
    elif args.model == 'KNeighbors':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_jobs=args.n_cores)
    elif args.model == 'SGD':
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(loss='log_loss', n_jobs=args.n_cores, random_state=seed)
    elif args.model == 'MLP':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(random_state=seed)
    elif args.model == 'SVC':
        from sklearn.svm import SVC
        model = SVC(probability=True, random_state=seed)
    
    # Grid search tuning
    if args.grid_search:
        print(f'Training {args.model} with GridSearch')
        from sklearn.model_selection import GridSearchCV
        model = GridSearchCV(model, param_grid=model_parameters[args.model], cv=args.cv_fold, n_jobs=args.n_cores)
        
    model.fit(X_train_data, Y_train_data.ravel())
    
    y_pred = model.predict(X_train_data)
    y_prob = model.predict_proba(X_train_data)

    labels = list(class_categories.keys())
    
    from sklearn.metrics import accuracy_score, f1_score, jaccard_score, log_loss, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve, classification_report, ConfusionMatrixDisplay
    
    accuracy = accuracy_score(Y_train_data, y_pred)
    f1 = f1_score(Y_train_data, y_pred)
    jaccard = jaccard_score(Y_train_data, y_pred)
    l_loss = log_loss(Y_train_data, y_pred)
    precision = precision_score(Y_train_data, y_pred)
    recall = recall_score(Y_train_data, y_pred)
    roc_auc = roc_auc_score(Y_train_data, y_prob[:, 1])
    roc = roc_curve(Y_train_data, y_prob[:, 1])
    
    print(classification_report(Y_train_data, y_pred))
    print(confusion_matrix(Y_train_data, y_pred))

    t_end = default_timer()
    print(f"Execution time: {(t_end - t_start)/60.0} minutes for {len(sar_filenames)} pair(s) of tile image(s)")
    
    wandb.login()
    # set up wandb logging
    wandb.init(project="sea-ice-classification")
    if args.name != "default":
        wandb.run.name = args.name
    wandb.sklearn.plot_classifier(model, X_train_data, X_train_data, Y_train_data, Y_train_data,
                                  y_pred, y_prob, labels, is_binary=is_binary, model_name=args.model)
    wandb.sklearn.plot_roc(Y_train_data, y_prob, labels)
    wandb.sklearn.plot_class_proportions(Y_train_data, Y_train_data, labels)
    wandb.sklearn.plot_precision_recall(Y_train_data, y_prob, labels)
    wandb.sklearn.plot_calibration_curve(model, X_train_data, Y_train_data, args.model)
    wandb.sklearn.plot_summary_metrics(model, X_train_data, Y_train_data, X_train_data, Y_train_data)
    # wandb.sklearn.plot_learning_curve(model, X, y)
    # wandb.log(args)
    wandb.log({"accuracy": accuracy,
               'f1': f1,
               'jaccard': jaccard,
               'log_loss': l_loss,
               'precision': precision,
               'recall': recall,
               'roc_auc': roc_auc,
               'roc': roc})
    
    if args.grid_search:
        wandb.log(model.best_params_)
    
    Path.mkdir(Path(f"scikit_models"), parents=True, exist_ok=True)
    dump(model, Path(f'scikit_models/{wandb.run.name}.joblib'))
    
    wandb.finish()
