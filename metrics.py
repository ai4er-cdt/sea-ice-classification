"""
AI4ER GTC - Sea Ice Classification
Functions to calculate mean and standard deviation
metrics for SAR images. These metrics are called in util.py
to normalise the SAR images.
"""
import numpy as np
from xarray.core.dataarray import DataArray
from tiling import load_raster
from pathlib import Path
from constants import chart_sar_pairs

def compute_metrics(array: DataArray) -> dict:
    
    """
    Computes the mean and the standard deviation of each band of a SAR image.
    In addition, computes mean and std of the ratio between HH and HV.

        Parameters:
            array (xarray.core.dataarray.DataArray): Original SAR image
            
        Returns:
            info (dict): Array metrics for future use
    """
    
    hh_hv = array[0] / (array[1] + 0.0001)
    hh_mean = np.nanmean(array[0].values)
    hv_mean = np.nanmean(array[1].values)
    angle_mean = np.nanmean(array[2].values)
    hh_hv_mean = np.nanmean(hh_hv.values)
    hh_std = np.nanstd(array[0].values)
    hv_std = np.nanstd(array[1].values)
    angle_std = np.nanstd(array[2].values)
    hh_hv_std = np.nanstd(hh_hv.values)
    
    info = {'hh_mean': hh_mean, 'hh_std': hh_std,
            'hv_mean': hv_mean, 'hv_std': hv_std,
            'angle_mean': angle_mean, 'angle_std': angle_std,
            'hh_hv_mean': hh_hv_mean, 'hh_hv_std': hh_hv_std}
    
    return info


def compute_overall_metrics(sar_folder: str, chart_sar_pairs: dict) -> dict:
    
    hh_total_sum = hv_total_sum = angle_total_sum = hh_hv_total_sum = total_pixels = 0

    for i, (_, sar_name, _) in enumerate(chart_sar_pairs):

        sar_image = load_raster(str(Path(f"{sar_folder}/{sar_name}.tif")), default_name="SAR Image")
        
        total_pixels += np.count_nonzero(~np.isnan(sar_image[0].values))
        
        hh_hv = sar_image[0] / (sar_image[1] + 0.0001)
        hh_total_sum += np.nansum(sar_image[0].values)
        hv_total_sum += np.nansum(sar_image[1].values)
        angle_total_sum += np.nansum(sar_image[2].values)
        hh_hv_total_sum += np.nansum(hh_hv.values)
        
    hh_mean = hh_total_sum / total_pixels
    hv_mean = hv_total_sum / total_pixels
    angle_mean = angle_total_sum / total_pixels
    hh_hv_mean = hh_hv_total_sum / total_pixels
    
    sqrd_diff = lambda array, mean: (array - mean)**2

    hh_std_sum = hv_std_sum = angle_std_sum = hh_hv_std_sum = 0
    
    for i, (_, sar_name, _) in enumerate(chart_sar_pairs):
        sar_image = load_raster(str(Path(f"{sar_folder}/{sar_name}.tif")), default_name="SAR Image")
        
        hh_hv = sar_image[0] / (sar_image[1] + 0.0001)    
        hh_std_sum += np.nansum(sqrd_diff(sar_image[0], hh_mean))
        hv_std_sum += np.nansum(sqrd_diff(sar_image[1], hv_mean))
        angle_std_sum += np.nansum(sqrd_diff(sar_image[2], angle_mean))
        hh_hv_std_sum += np.nansum(sqrd_diff(hh_hv, hh_hv_mean))
        
    hh_std = np.sqrt(hh_std_sum / total_pixels)
    hv_std = np.sqrt(hv_std_sum / total_pixels)
    angle_std = np.sqrt(angle_std_sum / total_pixels)
    hh_hv_std = np.sqrt(hh_hv_std_sum / total_pixels)
    
    info = {'hh_mean': hh_mean, 'hh_std': hh_std,
            'hv_mean': hv_mean, 'hv_std': hv_std,
            'angle_mean': angle_mean, 'angle_std': angle_std,
            'hh_hv_mean': hh_hv_mean, 'hh_hv_std': hh_hv_std}
    
    return info