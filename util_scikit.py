
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
from numpy import ndarray
from xarray.core.dataarray import DataArray

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


def load_sar(file_path: str, sar_band3: bool, parse_coordinates: bool=True) -> ndarray:
    
    sar = rxr.open_rasterio(file_path, parse_coordinates=parse_coordinates)
    band3_sar = define_band3(sar, sar_band3)
    normalized_raster = normalize_sar(band3_sar, sar_band3)
    
    return normalized_raster.values


def load_chart(file_path: str, class_categories: dict, parse_coordinates: bool=True, masked: bool=True, flip_vertically: bool=False) -> ndarray:
    
    chart = rxr.open_rasterio(file_path, parse_coordinates=parse_coordinates, masked=masked)
    if flip_vertically:
        chart.reindex(y=chart.y[::-1])
    new_raster = recategorize_chart(chart.values, class_categories)
    
    return new_raster

    
def crop_image(raster: DataArray, height_size: int, width_size: int) -> ndarray:
    _, height, width = raster.shape
    y_pad = (height-height_size) // 2
    x_pad = (width-width_size) // 2
    raster = raster[:, y_pad:y_pad+height_size, x_pad:x_pad+width_size]
    
    return raster
