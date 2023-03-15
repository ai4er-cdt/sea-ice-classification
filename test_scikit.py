"""
AI4ER GTC - Sea Ice Classification
Script for feeding test data into scikit-learn
classifiers saving the model output to wandb
"""
import os
import wandb
import numpy as np
import multiprocessing as mp
from pathlib import Path
from timeit import default_timer
from joblib import load
from constants import new_classes, model_parameters, chart_sar_pairs
from util_scikit import load_chart, load_sar, crop_image
from argparse import ArgumentParser, BooleanOptionalAction

if __name__ == "__main__":

    parser = ArgumentParser(description="Sea Ice Segmentation Test")
   
    # parser.add_argument("--username", type=str, help="wandb username")
    # parser.add_argument("--name", type=str, help="Name of wandb run")
    parser.add_argument("--model_name", type=str, help="path to the model")
    parser.add_argument("--pct_sample", default=0.1, type=float, help="Percent of images to use as sample")
    parser.add_argument("--load_parallel", action=BooleanOptionalAction, help='Whether to read tiles in parallel')
    parser.add_argument("--classification_type", default="binary", type=str,
                        choices=["binary", "ternary", "multiclass"], help="Type of classification task")
    parser.add_argument("--sar_band3", default="angle", type=str, choices=["angle", "ratio"],
                        help="Whether to use incidence angle or HH/HV ratio in third band")
    parser.add_argument("--sar_folder", default='sar', type=str, help="SAR input folder name")
    parser.add_argument("--chart_folder", default='chart', type=str, help="Ice Chart input folder name")
    parser.add_argument("--n_cores", default=-1, type=int, help="Number of jobs to run in parallel")
    parser.add_argument("--data_type", default='tile', type=str, choices=['tile', 'original'], help='Run the classifier on the tiles or the original images')
    parser.add_argument("--seed", default=0, type=int, help="Numpy random seed")
    args = parser.parse_args()
    
    t_start = default_timer()
    
    class_categories = new_classes[args.classification_type]
    n_classes = len(class_categories)
    sar_band3 = args.sar_band3
    is_binary = True if args.classification_type == 'binary' else False
    seed = np.random.seed(args.seed)
    
    # Function wrappers for parallel execution
    def load_sar_wrapper(file_path: str):
        return load_sar(file_path, sar_band3)
        
    def load_chart_wrapper(file_path: str):
        return load_chart(file_path, class_categories)
    
    def load_chart_wrapper_vertical(file_path: str):
        return load_chart(file_path, class_categories, flip_vertically=args.flip_vertically)

    # standard input dirs
    if args.data_type == 'tile':
        input_folder = Path(open("tile.config").read().strip())
        sar_folder = f"{input_folder}/test/{args.sar_folder}"
        chart_folder = f"{input_folder}/test/{args.chart_folder}"

        sar_filenames = os.listdir(sar_folder)
        sar_filenames.sort()
        chart_filenames = os.listdir(chart_folder)
        chart_filenames.sort()
        sar_filenames = [os.path.join(sar_folder, x) for x in sar_filenames]
        chart_filenames = [os.path.join(chart_folder, x) for x in chart_filenames]
        
    elif args.data_type == 'original':
        input_folder = Path(open("ftp.config").read().strip())
        sar_folder = f"{input_folder}/test_ims/original_sar_images/dual_band_images"
        chart_folder = f"{input_folder}/test_ims/rasterised_ice_charts/rasterised_shapefiles"
        chart_ext = "tiff"
        sar_ext = "tif"
        
        sar_filenames = [os.path.join(sar_folder, f'{sar}.{sar_ext}') for (_, sar, _) in chart_sar_pairs]
        chart_filenames = [os.path.join(chart_folder, f'{chart}.{chart_ext}') for (chart, _, _) in chart_sar_pairs]
    
    # Sample tiles according to argsparse
    if args.sample == 'True':
        assert 0 < args.pct_sample <= 1
        n_sample = int(len(sar_filenames) * args.pct_sample)
        sample_n = np.random.randint(len(sar_filenames), size=(n_sample))
        sar_filenames = [sar_filenames[i] for i in sample_n]
        chart_filenames = [chart_filenames[i] for i in sample_n]

    # Standard or parallel loading of tiles
    if args.load_parallel:
        print('Loading tiles in parallel')
        cores = mp.cpu_count() if args.n_cores == -1 else args.n_cores
        mp_pool = mp.Pool(cores)
        
        test_x_lst = mp_pool.map(load_sar_wrapper, sar_filenames)
        if args.data_type == 'original' and args.flip_vertically:
            test_y_lst = mp_pool.map(load_chart_wrapper_vertical, chart_filenames)
        else:
            test_y_lst = mp_pool.map(load_chart_wrapper, chart_filenames)
        
        mp_pool.close()
    else:
        print(f'Loading {len(sar_filenames)} tiles')
        test_x_lst = [load_sar(sar, sar_band3=sar_band3) for sar in sar_filenames]
        test_y_lst = [load_chart(chart, class_categories, flip_vertically=args.flip_vertically) for chart in chart_filenames]
        
    # Crop tiles to the smallest size from the original SAR/Ice charts        
    if args.data_type == 'original':
        height_min = 100000000
        width_min = 100000000
        height_max = 0
        width_max = 0
        for chart in test_y_lst:
            shape = chart.shape
            if shape[1] < height_min:
                height_min = shape[1]
            if shape[2] < width_min:
                width_min = shape[2]
        
        dim_min = min([height_min, width_min])
                
        test_x_lst = [crop_image(sar, height_min, width_min) for sar in test_x_lst]
        test_y_lst = [crop_image(chart, height_min, width_min) for chart in test_y_lst]


    # Stack list of images as ndarray
    test_x = np.stack(test_x_lst)
    test_y = np.stack(test_y_lst)

    # Reorder dimensions
    X_test_data = np.moveaxis(test_x, 1, -1).reshape(-1, 3)
    Y_test_data = np.moveaxis(test_y, 1, -1).reshape(-1, 1)
    
    # Intel optimizer for Intel machines
    from sklearnex import patch_sklearn
    patch_sklearn()

    # wandb logging
    # wandb.init(id=args.name, project="sea-ice-classification", resume="must")
    # api = wandb.Api()
    # run = api.run(f"{args.username}/sea-ice-classification/{args.name}")
    
    model = load(Path(f'scikit_models/{args.model_name}.joblib'))
    
    model.fit(X_test_data, Y_test_data.ravel())
    
    y_pred = model.predict(X_test_data)
    y_prob = model.predict_proba(X_test_data)

    labels = list(class_categories.keys())
    
    # Sklearn metrics
    from sklearn.metrics import accuracy_score, f1_score, jaccard_score, log_loss, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve, r2_score, mean_absolute_error, mean_squared_error, classification_report, ConfusionMatrixDisplay
    
    test_jaccard = jaccard_score(Y_test_data, y_pred, average='macro', labels=labels)
    test_accuracy = accuracy_score(Y_test_data, y_pred)
    test_micro_precision = precision_score(Y_test_data, y_pred, average="micro", labels=labels)
    test_macro_precision = precision_score(Y_test_data, y_pred, average="macro", labels=labels)
    test_weighted_precision = precision_score(Y_test_data, y_pred, average="weighted", labels=labels)
    test_micro_recall = recall_score(Y_test_data, y_pred, average="micro", labels=labels)
    test_macro_recall = recall_score(Y_test_data, y_pred, average="macro", labels=labels)
    test_weighted_recall = recall_score(Y_test_data, y_pred, average="weighted", labels=labels)
    test_micro_f1 = f1_score(Y_test_data, y_pred, average="micro", labels=labels)
    test_macro_f1 = f1_score(Y_test_data, y_pred, average="macro", labels=labels)
    test_weighted_f1 = f1_score(Y_test_data, y_pred, average="weighted", labels=labels)
    test_mse = mean_squared_error(Y_test_data, y_pred)
    test_rmse = mean_squared_error(Y_test_data, y_pred, squared=False)
    test_mae = mean_absolute_error(Y_test_data, y_pred)
    # test_l_loss = log_loss(Y_test_data, y_pred, labels=labels)    
    # test_roc_auc = roc_auc_score(Y_test_data, y_prob[:, 1], labels=labels, multi_class='ovr')
    # test_roc = roc_curve(Y_test_data, y_prob[:, 1])
    test_r2 = r2_score(Y_test_data, y_pred)
    
    metrics_dict = {'test_jaccard': test_jaccard, 'test_accuracy': test_accuracy,
                    'test_micro_precision': test_micro_precision, 'test_macro_precision': test_macro_precision,
                    'test_weighted_precision': test_weighted_precision, 'test_micro_recall': test_micro_recall,
                    'test_macro_recall': test_macro_recall, 'test_weighted_recall': test_weighted_recall,
                    'test_micro_f1': test_micro_f1, 'test_macro_f1': test_macro_f1, 'test_weighted_f1': test_weighted_f1,
                    'test_mse': test_mse, 'test_rmse': test_rmse, 'test_mae': test_mae, 
                    # 'test_log_loss': test_l_loss, 'test_roc_auc': test_roc_auc, 'test_roc': test_roc,
                    'test_r2': test_r2}
    
    print(classification_report(Y_test_data, y_pred))
    print(confusion_matrix(Y_test_data, y_pred))

    t_end = default_timer()
    print(f"Execution time: {(t_end - t_start)/60.0} minutes for {len(sar_filenames)} pair(s) of tile image(s)")

    import json
    
    with open(Path(f'test_{args.model_name}.json'), 'w') as f:
        json.dump(metrics_dict, f)