
"""
AI4ER GTC - Sea Ice Classification
Functions for loading and tiling of raster files
"""

import re
import xarray as xr
import rioxarray as rxr
from xarray.core.dataarray import DataArray
from pathlib import Path
from timeit import default_timer 


def load_raster(file_path: str, parse_coordinates: bool = True, masked: bool = True, default_name: str = None,
                crs: int = 4326) -> DataArray:
    
    """
    Loads and returns xarray.core.dataarray.DataArray for a given raster file

        Parameters:
            file_path (str): Path to file
            parse_coordinates (bool): Parses the coordinates of the file, if any
            masked (bool): Reads raster as a mask
            default_name (str): Name for the array
            crs (int): Coordinate Reference System

        Returns:
            raster (xarray.core.dataarray.DataArray): DataArray from file
    """

    raster = rxr.open_rasterio(Path(file_path), parse_coordinates=parse_coordinates, masked=masked,
                               default_name=default_name)

    assert type(raster) == xr.core.dataarray.DataArray  # Makes sure the data structure is DataArray
    raster.rio.write_crs(crs, inplace=True)  # Modifies CRS of the raster

    return raster


def tile_raster(sar_image: DataArray, ice_chart: DataArray, output_folder: str, basename: str, region_prefix: str,
                size_x: int = 256, size_y: int = 256, start_x: int = 0, start_y: int = 0,
                end_x: int = None, end_y: int = None, stride_x: int = 128, stride_y: int = 128,
                nan_threshold: float = 1.0) -> tuple[int, int]:
    
    """
    Slices a given pair of source images using a moving window
    Outputs valid sar and ice chart tiles to disk
    Also outputs a binary mask corresponding to each ice chart tile (1 = ice, 0 = water)
    Invalid tile pairs are skipped, e.g. due to NaN or values out of range

        Parameters:
            sar_image (xarray.core.dataarray.DataArray): Original SAR image to be tiled
            ice_chart (xarray.core.dataarray.DataArray): Corresponding rasterised ice chart to be tiled
            output_folder (str): Path to folder where tiled images will be saved
            basename (str): Base name for saved files
            region_prefix (str): 2-character prefix indicating geographical region of the source files
                e.g. 'WS' = (East) Weddell Sea region, 'AP' = Antarctic Peninsula region
            size_x (int): Tile size in the horizontal axis
            size_y (int): Tile size in the vertical axis
            start_x (int): Starting coordinate for tiling in the horizontal axis
            start_y (int): Starting coordinate for tiling in the vertical axis
            end_x (int): Final coordinate for tiling in the horizontal axis
            end_y (int): Final coordinate for tiling in the vertical axis
            stride_x (int): Stride of the moving window in the horizontal axis
            stride_y (int): Stride of the moving window in the vertical axis
            nan_threshold (float): number in [0,1]

        Returns:
            img_n (int): Number of image / tile pairs generated
            discared_tiles (int): Number of tile pairs discarded
    """

    # Output config
    sar_subfolder = "sar"
    chart_subfolder = "chart"
    binary_subfolder = "binary_chart"
    sar_prefix = "SAR"
    chart_prefix = "CHART"
    binary_prefix = "BINARY_CHART"
    output_ext = "tiff"

    # Makes sure some parameters have default values according to input
    end_x = sar_image.shape[2] if end_x is None else end_x
    end_y = sar_image.shape[1] if end_y is None else end_y
    stride_x = size_x if stride_x is None else stride_x
    stride_y = size_y if stride_y is None else stride_y

    # Checkpoint of parameters
    assert len(region_prefix) == 2
    assert 0 < size_x <= sar_image.shape[2] // 2
    assert 0 < size_y <= sar_image.shape[1] // 2
    assert 0 <= start_x <= sar_image.shape[2]
    assert 0 <= start_y <= sar_image.shape[1]
    assert 0 < end_x <= sar_image.shape[2]
    assert 0 < end_y <= sar_image.shape[1]
    assert 0 < stride_x <= size_x
    assert 0 < stride_y <= size_y
    assert 0.0 <= nan_threshold <= 1.0

    # Create output dirs if they don't exist
    Path.mkdir(Path(f"{output_folder}/{sar_subfolder}"), parents=True, exist_ok=True)
    Path.mkdir(Path(f"{output_folder}/{chart_subfolder}"), parents=True, exist_ok=True)
    Path.mkdir(Path(f"{output_folder}/{binary_subfolder}"), parents=True, exist_ok=True)

    img_n = 0  # Counter for image pairs generated (+1 for file naming convention)
    discarded_tiles = 0  # Counter for discarded tile pairs

    # Iterates over rows and columns of both images according to input parameters
    for row in range(start_y, end_y, stride_y):
        for col in range(start_x, end_x, stride_x):

            # Indexes ice chart and SAR image for the current tile
            sub_sar = sar_image[:, row:row + size_y, col:col + size_x]
            sub_chart = ice_chart[:, row:row + size_y, col:col + size_x]

            # Checks if the current tile has the same shape as the parameters, if not it skips to the next tile
            if (sub_sar.shape[1] * sub_sar.shape[2] != size_x * size_y) or (
                    sub_chart.shape[1] * sub_chart.shape[2] != size_x * size_y):
                continue

            # NaN Check: Skip to next pair if too many NaNs in either tile
            # Range Check: Make sure all values are between 0 and 100
            # TBC: Only checks first two layers of SAR tile (i.e. ignores incidence angle)
            if (sub_chart.isnull().sum().values / (size_x * size_y) >= nan_threshold
                    or sub_sar[0, :, :].isnull().sum().values / (size_x * size_y) >= nan_threshold
                    or sub_sar[1, :, :].isnull().sum().values / (size_x * size_y) >= nan_threshold
                    or sub_chart[0, :, :].values.min() < 0
                    or sub_chart[0, :, :].values.max() > 100
            ):
                discarded_tiles += 1
                continue

            # Make a copy of the chart and set to binary classification objective
            sub_binary = sub_chart.copy()
            sub_binary.values[sub_binary.values <= 1] = 0  # these are water pixels
            sub_binary.values[sub_binary.values > 1] = 1  # these are ice pixels

            # Majority of filename is common to both sar and chart tiles
            file_n = "{:0>5}".format(img_n + 1)
            common_fname = f"{region_prefix}_{basename}_{file_n}_[{col},{row}]_{size_x}x{size_y}.{output_ext}"

            # Separate by subfolder and prefix
            pathout_sar = f"{output_folder}/{sar_subfolder}/{sar_prefix}_{common_fname}"
            pathout_chart = f"{output_folder}/{chart_subfolder}/{chart_prefix}_{common_fname}"
            pathout_binary = f"{output_folder}/{binary_subfolder}/{binary_prefix}_{common_fname}"

            # Save to disk
            sub_sar.rio.to_raster(Path(pathout_sar))
            sub_chart.rio.to_raster(Path(pathout_chart))
            sub_binary.rio.to_raster(Path(pathout_binary))

            img_n += 1

    print(f"Number of image pairs generated: {img_n}")
    print(f"Number of discarded tile pairs: {discarded_tiles}")

    return img_n, discarded_tiles


if __name__ == "__main__":

    """
    Runs loading and tiling of image pairs

    The following files must exist in the working directory: 

    data_path.config: 
        Specifies the path of the local data folder
        this will vary from system to system

    constants.py: 
        Contains a list of the image pairs to be processed
        as well as associated metadata such as geographical region

    """

    # User config
    n_pairs_to_process = 2
    output_folder = "../Tiled_images"
    resolution = 256
    stride = 128
    flip_charts = True  # ice charts may need vertical flip before tiling

    # Standard config 
    base_folder = open("data_path.config").read()
    chart_folder = Path(f"{base_folder}/FTP_data/rasterised_shapefiles")  
    sar_folder = Path(f"{base_folder}/FTP_data/dual_band_images")
    chart_ext = "tiff"; sar_ext = "tif"

    # Prepare to run
    t_start = default_timer()
    total_img = 0; 
    total_discarded = 0  

    # Run loading and tiling of image pairs
    from constants import chart_sar_pairs
    for i, (chart_name, sar_name, region) in enumerate(chart_sar_pairs):
        if i >= n_pairs_to_process:
            break
        chart_image = load_raster(Path(f"{chart_folder}/{chart_name}.{chart_ext}"), default_name="Ice Chart")
        if flip_charts == True:
            chart_image = chart_image.reindex(y=chart_image.y[::-1])  # flip vertically
        sar_image = load_raster(Path(f"{sar_folder}/{sar_name}.{sar_ext}"), default_name="SAR Image")
        name_extract = re.findall("H_[0-9]{8}T", sar_name)[0][2:10]  # use sar date as identifier for all outputs
        print(f"Tiling {name_extract} ...")
        img_n, discarded_tiles = tile_raster(sar_image, chart_image, Path(output_folder), name_extract, region, size_x=resolution, size_y=resolution, stride_x=stride, stride_y=stride)
        total_img += img_n; total_discarded += discarded_tiles

    print("TILING COMPLETE\n")
    print(f"Total image pairs generated: {total_img}")
    print(f"Total discarded tile pairs: {total_discarded}")
    print(f"Proportion discarded: {total_discarded/(total_img+total_discarded)}")

    t_end = default_timer()
    print(f"Execution time: {(t_end - t_start)/60.0} minutes for {n_pairs_to_process} pairs of source images")
