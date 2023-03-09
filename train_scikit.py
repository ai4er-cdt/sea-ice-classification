
import os
import wandb
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
from xarray.core.dataarray import DataArray
from pathlib import Path
from timeit import default_timer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from constants import new_classes, chart_sar_pairs
from tiling import load_raster
from argparse import ArgumentParser

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


if __name__ == '__main__':
    
    parser = ArgumentParser(description="Sea Ice Random Forest Train")
    parser.add_argument("--name", default="default", type=str, help="Name of wandb run")
    parser.add_argument("--sample", default="False", type=str, choices=['True', 'False'], help="Run a sample of the dataset")
    parser.add_argument("--classification_type", default="binary", type=str,
                        choices=["binary", "ternary", "multiclass"], help="Type of classification task")
    parser.add_argument("--sar_band3", default="angle", type=str, choices=["angle", "ratio"],
                        help="Whether to use incidence angle or HH/HV ratio in third band")
    parser.add_argument("--flip_vertically", default="True", type=str, choices=["True", "False"],
                        help="Whether to flip an ice chart vertically to match the SAR coordinates")
    parser.add_argument("--sar_folder", default='sar', type=str, help="SAR output folder name")
    parser.add_argument("--chart_folder", default='chart', type=str, help="Ice Chart output folder name")
    parser.add_argument("--n_cores", default=-1, type=int, help="Number of jobs to run in parallel")
    args = parser.parse_args()
    
    t_start = default_timer()
    # standard input dirs
    output_folder = Path(open("tile.config").read().strip())
    sar_folder = f"{output_folder}/{args.sar_folder}"
    chart_folder = f"{output_folder}/{args.chart_folder}"
    class_categories = new_classes[args.classification_type]
    n_classes = len(class_categories)
    
# Use original images as input ---------------------------
    # # image_folder = open("ftp.config").read().strip()
    # chart_folder = f"{image_folder}/rasterised_shapefiles"
    # sar_folder = f"{image_folder}/dual_band_images"

    # chart_ext = "tiff"
    # sar_ext = "tif"
    # # Validation files
    # chart_val_names = ['20190313', '20200117']
    # sar_val_names = ['S1B_EW_GRDM_1SDH_20190313T232241_20190313T232345_015342_01CB99_7DC1', 'S1B_EW_GRDM_1SDH_20200117T220139_20200117T220243_019862_02590A_7B65']

    # # get train file lists
    # sar_train_files = [os.path.join(sar_folder, f'{sar}.{sar_ext}') for (_, sar, _) in chart_sar_pairs if sar not in sar_val_names]
    # chart_train_files = [os.path.join(chart_folder, f'{chart}.{chart_ext}') for (chart, _, _) in chart_sar_pairs if chart not in chart_val_names]    
    
    # # get validation file lists
    # sar_val_files = [os.path.join(sar_folder, f'{sar}.{sar_ext}') for sar in chart_sar_pairs if sar in sar_val_names]
    # chart_val_files = [os.path.join(chart_folder, f'{chart}.{chart_ext}') for chart in chart_sar_pairs if chart in chart_val_names]    

    # # Stack DataArrays
    # train_x_lst = [normalize_sar(define_band3(rxr.open_rasterio(x, parse_coordinates=True), sar_band3=args.sar_band3), sar_band3=args.sar_band3).values for x in sar_train_files]
    # val_x_lst = [normalize_sar(define_band3(rxr.open_rasterio(x, parse_coordinates=True), sar_band3=args.sar_band3), sar_band3=args.sar_band3).values for x in sar_val_files]

    # if args.flip_vertically == 'True':

    #     train_y_lst = [rxr.open_rasterio(y, parse_coordinates=True, masked=True) for y in chart_train_files]
    #     train_y_lst = [recategorize_chart(chart.reindex(y=chart.y[::-1]).values, class_categories) for chart in train_y_lst]
    #     val_y_lst = [rxr.open_rasterio(y, parse_coordinates=True, masked=True) for y in chart_val_files]
    #     val_y_lst = [recategorize_chart(chart.reindex(y=chart.y[::-1]).values, class_categories) for chart in val_y_lst]
    
    # elif args.flip_vertically == 'False':
    
    #     train_y_lst = [recategorize_chart(rxr.open_rasterio(y, parse_coordinates=True, masked=True).values, class_categories) for y in chart_train_files]
    #     val_y_lst = [recategorize_chart(rxr.open_rasterio(y, parse_coordinates=True, masked=True).values, class_categories) for y in chart_val_files]
# ---------------------------------------------------
    sar_filenames = os.listdir(sar_folder)
    chart_filenames = os.listdir(chart_folder)
    
    if args.sample == 'True':
        np.random.seed(0)
        sample_n = np.random.randint(len(sar_filenames), size=(50))
        sar_filenames = [sar_filenames[i] for i in sample_n]
        chart_filenames = [chart_filenames[i] for i in sample_n]

    train_x_lst = [normalize_sar(define_band3(rxr.open_rasterio(os.path.join(sar_folder, x), parse_coordinates=True), sar_band3=args.sar_band3), sar_band3=args.sar_band3).values for x in sar_filenames]
    train_y_lst = [recategorize_chart(rxr.open_rasterio(os.path.join(chart_folder, x), parse_coordinates=True, masked=True).values, class_categories) for x in chart_filenames]   

    train_x = np.stack(train_x_lst)
    train_y = np.stack(train_y_lst)
    # val_x = np.stack(val_x_lst)
    # val_y = np.stack(val_y_lst)

    # Reorder dimensions
    x_train = np.moveaxis(train_x, 1, -1)
    X_train_data = x_train.reshape(-1, 3)
    y_train = np.moveaxis(train_y, 1, -1)
    Y_train_data = y_train.reshape(-1, 1)

    # x_val = np.moveaxis(val_x, 1, -1)
    # X_val_data = x_val.reshape(-1, 3)
    # y_val = np.moveaxis(val_y, 1, -1)
    # Y_val_data = y_val.reshape(-1, 1)

    import multiprocessing

    n_cores = multiprocessing.cpu_count()
    # RF
    rf = RandomForestClassifier(n_jobs=n_cores)
    rf.fit(X_train_data, Y_train_data.ravel())
    
    rf_pred = rf.predict(X_train_data)
    y_probas = rf.predict_proba(X_train_data)

    labels = list(class_categories.keys())
    
    print(f"Accuracy: {accuracy_score(Y_train_data, rf_pred)*100}")

    print(classification_report(Y_train_data, rf_pred))
    print(confusion_matrix(Y_train_data,rf_pred))

    t_end = default_timer()
    print(f"Execution time: {(t_end - t_start)/60.0} minutes for {len(sar_filenames)} pair(s) of tile image(s)")
    
    wandb.login()
    # set up wandb logging
    wandb.init(project="sea-ice-classification")
    if args.name != "default":
        wandb.run.name = args.name
    wandb.sklearn.plot_classifier(rf, 
                              X_train_data, X_train_data, 
                              Y_train_data, Y_train_data, 
                              rf_pred, y_probas, 
                              labels, 
                              is_binary=True, 
                              model_name='RandomForest')

    wandb.finish()

