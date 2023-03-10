"""
AI4ER GTC - Sea Ice Classification
Script for feeding training and validation data into 
scikit-learn classifiers saving the model output to wandb
"""
import os
import wandb
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
import multiprocessing as mp
from xarray.core.dataarray import DataArray
from pathlib import Path
from timeit import default_timer
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from constants import new_classes, model_parameters
from argparse import ArgumentParser, BooleanOptionalAction


def define_band3(sar: DataArray, sar_band3: str = 'angle') -> DataArray:

    if sar_band3 == "ratio":

        band1 = sar.sel(band=1)
        band2 = sar.sel(band=2)
        band3 = sar.sel(band=3)
        band3.values = (band1.values / (band2.values + 0.0001))
        # Update the values of band 3 to the HH/HV ratio
        # Note: do not need to update the CRS or X/Y dimensions because they are the same as band 1 and 2
        sar.loc[dict(band=3)] = band3

    return sar
    

def normalize_sar(sar: DataArray, sar_band3: str = 'angle') -> DataArray:

    metrics_df = pd.read_csv("metrics.csv", delimiter=",")
    hh_mean = metrics_df.iloc[0]["hh_mean"]
    hh_std = metrics_df.iloc[0]["hh_std"]
    hv_mean = metrics_df.iloc[0]["hv_mean"]
    hv_std = metrics_df.iloc[0]["hv_std"]
    angle_mean = metrics_df.iloc[0]["angle_mean"]
    angle_std = metrics_df.iloc[0]["angle_std"]
    ratio_mean = metrics_df.iloc[0]["hh_hv_mean"]
    ratio_std = metrics_df.iloc[0]["hh_hv_std"]

    if sar_band3 == "angle":
        band3_mean = angle_mean
        band3_std = angle_std
    else:
        band3_mean = ratio_mean
        band3_std = ratio_std

    sar[0] = (sar[0] - hh_mean) / hh_std
    sar[1] = (sar[1] - hv_mean) / hv_std
    sar[2] = (sar[2] - band3_mean) / band3_std

    return sar


def recategorize_chart(chart: DataArray, class_categories: dict) -> DataArray:
    
    if class_categories is not None:
        for key, value in class_categories.items():
            chart[np.isin(chart, value)] = key

    return chart


def load_sar(file_path: str, sar_band3: bool, parse_coordinates: bool=True):
    
    sar = rxr.open_rasterio(file_path, parse_coordinates=parse_coordinates)
    band3_sar = define_band3(sar, sar_band3)
    normalized_raster = normalize_sar(band3_sar, sar_band3)
    
    return normalized_raster.values


def load_chart(file_path: str, class_categories: dict, parse_coordinates: bool=True, masked: bool=True):
    
    chart = rxr.open_rasterio(file_path, parse_coordinates=parse_coordinates, masked=masked)
    new_raster = recategorize_chart(chart.values, class_categories)
    
    return new_raster


if __name__ == '__main__':
    
    parser = ArgumentParser(description="Sea Ice Random Forest Train")
    parser.add_argument("--name", default="default", type=str, help="Name of wandb run")
    parser.add_argument("--sample", default="False", type=str, choices=['True', 'False'], help="Run a sample of the dataset")
    parser.add_argument("--load_parallel", action=BooleanOptionalAction, help='Whether to read tiles in parallel')
    parser.add_argument("--classification_type", default="binary", type=str,
                        choices=["binary", "ternary", "multiclass"], help="Type of classification task")
    parser.add_argument("--sar_band3", default="angle", type=str, choices=["angle", "ratio"],
                        help="Whether to use incidence angle or HH/HV ratio in third band")
    parser.add_argument("--sar_folder", default='sar', type=str, help="SAR output folder name")
    parser.add_argument("--chart_folder", default='chart', type=str, help="Ice Chart output folder name")
    parser.add_argument("--model", default='RandomForest', type=str, 
                        choices=['RandomForest', 'DecisionTree', 'KNeighbors', 'SGD', 'MLP'], help="Classification model to use")
    parser.add_argument("--grid_search", action=BooleanOptionalAction, help='Wether to perform grid search cross-validation')
    parser.add_argument("--cv_fold", default=5, type=int, help="Number of folds for cross-validation")
    parser.add_argument("--n_cores", default=-1, type=int, help="Number of jobs to run in parallel")
    args = parser.parse_args()
    
    t_start = default_timer()
    
    # standard input dirs
    output_folder = Path(open("tile.config").read().strip())
    sar_folder = f"{output_folder}/{args.sar_folder}"
    chart_folder = f"{output_folder}/{args.chart_folder}"
    class_categories = new_classes[args.classification_type]
    n_classes = len(class_categories)
    is_binary = True if args.classification_type == 'binary' else False
    seed = np.random.seed(0)

    sar_filenames = os.listdir(sar_folder)
    sar_filenames = [os.path.join(sar_folder, x) for x in sar_filenames]
    chart_filenames = os.listdir(chart_folder)
    chart_filenames = [os.path.join(chart_folder, x) for x in chart_filenames]
    
    if args.sample == 'True':
        
        sample_n = np.random.randint(len(sar_filenames), size=(50))
        sar_filenames = [sar_filenames[i] for i in sample_n]
        chart_filenames = [chart_filenames[i] for i in sample_n]
        
    if args.load_parallel:
        cores = mp.cpu_count() if args.n_cores == -1 else args.n_cores
        mp_pool = mp.Pool(cores)
        
        def load_sar_wrapper(file_path: str):
            return load_sar(file_path, args.sar_band3)
        
        def load_chart_wrapper(file_path: str):
            return load_chart(file_path, class_categories)
        
        train_x_lst = mp_pool.map(load_sar_wrapper, sar_filenames)
        train_y_lst = mp_pool.map(load_chart_wrapper, chart_filenames)
        
        mp_pool.close()
    else:
        train_x_lst = [normalize_sar(define_band3(rxr.open_rasterio(x, parse_coordinates=True), sar_band3=args.sar_band3), sar_band3=args.sar_band3).values for x in sar_filenames]
        train_y_lst = [recategorize_chart(rxr.open_rasterio(x, parse_coordinates=True, masked=True).values, class_categories) for x in chart_filenames]   

    train_x = np.stack(train_x_lst)
    train_y = np.stack(train_y_lst)

    # Reorder dimensions
    x_train = np.moveaxis(train_x, 1, -1)
    X_train_data = x_train.reshape(-1, 3)
    y_train = np.moveaxis(train_y, 1, -1)
    Y_train_data = y_train.reshape(-1, 1)

    # Models
    print(f'Training {args.model}')
    if args.model == 'RandomForest':
        model = RandomForestClassifier(n_jobs=args.n_cores, random_state=seed)
    elif args.model == 'DecisionTree':
        from sklearn.tree import  DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=seed)
    elif args.model == 'KNeighbors':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_jobs=args.n_cores)
    elif args.model == 'SGD':
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(n_jobs=args.n_cores, random_state=seed)
    elif args.model == 'MLP':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(random_state=seed)
    
    # Grid search tuning
    if args.grid_search:
        print(f'Training {args.model} with GridSearch')
        from sklearn.model_selection import GridSearchCV
        model = GridSearchCV(model, param_grid=model_parameters[args.model], cv=args.cv_fold, n_jobs=args.n_cores)
        
    model.fit(X_train_data, Y_train_data.ravel())
    
    y_pred = model.predict(X_train_data)
    y_prob = model.predict_proba(X_train_data)

    labels = list(class_categories.keys())
    
    print(f"Accuracy: {accuracy_score(Y_train_data, y_pred)*100}")

    print(classification_report(Y_train_data, y_pred))
    print(confusion_matrix(Y_train_data,y_pred))

    t_end = default_timer()
    print(f"Execution time: {(t_end - t_start)/60.0} minutes for {len(sar_filenames)} pair(s) of tile image(s)")
    
    wandb.login()
    # set up wandb logging
    wandb.init(project="sea-ice-classification")
    if args.name != "default":
        wandb.run.name = args.name
    wandb.sklearn.plot_classifier(model, X_train_data, X_train_data, Y_train_data, Y_train_data,
                                  y_pred, y_prob, labels, is_binary=is_binary, model_name=args.model)
    
    Path.mkdir(Path(f"scikit_models"), parents=True, exist_ok=True)
    dump(model, Path(f'scikit_models/{wandb.run.name}.joblib') )
    
    wandb.finish()
