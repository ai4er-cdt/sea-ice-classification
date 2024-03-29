"""
AI4ER GTC - Sea Ice Classification
Functions for loading and tiling of raster files
"""
import re
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from xarray.core.dataarray import DataArray
from datetime import datetime
from timeit import default_timer
from argparse import ArgumentParser
from pathlib import Path


def load_raster(file_path: str, parse_coordinates: bool = True, masked: bool = True, default_name: str = None,
                crs: int = 4326) -> DataArray:
    
    """
    Loads and returns xarray.core.dataarray.DataArray for a given raster file

        Parameters:
            file_path (str): Path to raster file
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
                end_x: int = None, end_y: int = None, stride_x: int = 128, stride_y: int = 128, nan_threshold: float = 0.0,
                sar_band3: bool = True, sar_subfolder: str = "sar", sar_band3_subfolder: str = "sar_band3", chart_subfolder: str = "chart") -> tuple[int, int, list]:
    
    """
    Slices a given pair of source images using a moving window
    Outputs valid sar and ice chart tiles to disk
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
            sar_band3 (bool): Whether to save a separate file with HH/HV as band 3
            sar_subfolder (str): SAR output folder name
            sar_band3_subfolder (str): SAR band3 output folder name
            chart_subfolder (str): Ice Chart output folder name
        Returns:
            img_n (int): Number of image / tile pairs generated
            discared_tiles (int): Number of tile pairs discarded
    """

    # Output config
    sar_prefix = "SAR"
    chart_prefix = "CHART"
    output_ext = "tiff"

    # Makes sure some parameters have default values according to input
    end_x = sar_image.shape[2] if end_x is None else end_x
    end_y = sar_image.shape[1] if end_y is None else end_y
    stride_x = size_x if stride_x is None else stride_x
    stride_y = size_y if stride_y is None else stride_y

    # Checkpoint of parameters
    assert len(region_prefix) == 2
    assert 0 < size_x <= sar_image.shape[2]
    assert 0 < size_y <= sar_image.shape[1]
    assert 0 <= start_x <= sar_image.shape[2]
    assert 0 <= start_y <= sar_image.shape[1]
    assert 0 < end_x <= sar_image.shape[2]
    assert 0 < end_y <= sar_image.shape[1]
    assert 0 < stride_x <= size_x
    assert 0 < stride_y <= size_y
    assert 0.0 <= nan_threshold <= 1.0

    # Create output dirs if they don't exist
    Path.mkdir(Path(f"{output_folder}/{sar_subfolder}"), parents=True, exist_ok=True)
    Path.mkdir(Path(f"{output_folder}/{sar_band3_subfolder}"), parents=True, exist_ok=True)
    Path.mkdir(Path(f"{output_folder}/{chart_subfolder}"), parents=True, exist_ok=True)

    img_n = 0  # Counter for image pairs generated (+1 for file naming convention)
    discarded_tiles = 0  # Counter for discarded tile pairs
    info_lst = []  # Empty list to be filled with each tile info

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
            if (sub_chart.isnull().sum().values / (size_x * size_y) > nan_threshold
                    or sub_sar[0, :, :].isnull().sum().values / (size_x * size_y) > nan_threshold
                    or sub_sar[1, :, :].isnull().sum().values / (size_x * size_y) > nan_threshold
                    or sub_sar[2, :, :].isnull().sum().values / (size_x * size_y) > nan_threshold
                    or sub_chart[0, :, :].values.min() < 0
                    or sub_chart[0, :, :].values.max() > 100
            ):
                discarded_tiles += 1
                continue

            # Majority of filename is common to both sar and chart tiles
            file_n = "{:0>5}".format(img_n + 1)
            common_fname = f"{region_prefix}_{basename}_{file_n}_[{col},{row}]_{size_x}x{size_y}.{output_ext}"

            # Separate by subfolder and prefix
            pathout_sar = f"{output_folder}/{sar_subfolder}/{sar_prefix}_{common_fname}"
            pathout_sar_band3 = f"{output_folder}/{sar_band3_subfolder}/{sar_prefix}_{common_fname}"
            pathout_chart = f"{output_folder}/{chart_subfolder}/{chart_prefix}_{common_fname}"

            # Save tile info in a dictionary
            unique, counts = np.unique(sub_chart, return_counts=True)
            info = dict(zip(unique.astype('str'), counts))
            info['region'] = region_prefix
            info['basename'] = basename
            info['file_n'] = file_n
            info['size'] = size_x
            info['col'] = col
            info['row'] = row
            info_lst.append(info)

            # Save to disk
            sub_sar.rio.to_raster(Path(pathout_sar))
            sub_chart.rio.to_raster(Path(pathout_chart))

            if sar_band3 == 'True':
                # Additionally, update band 3 values in tiled sar images and save to a new folder
                # Calculate the ratio of the HH/HV bands
                band1 = sub_sar.sel(band=1)
                band2 = sub_sar.sel(band=2)
                band3 = sub_sar.sel(band=3)
                band3.values = (band1.values / band2.values)
                # Update the values of band 3 to the HH/HV ratio
                # Note: do not need to update the CRS or X/Y dimensions because they are the same as band 1 and 2
                sub_sar.loc[dict(band=3)] = band3
                # for checking only --> print(sub_sar.sel(band=3).values)
                # Save the updated SAR image to disk
                sub_sar.rio.to_raster(Path(pathout_sar_band3), overwrite=True)

            img_n += 1

    print(f"Number of image pairs generated: {img_n}")
    print(f"Number of discarded tile pairs: {discarded_tiles}")

    return img_n, discarded_tiles, info_lst


def create_tile_info_dataframe(lst: list, output_folder: str) -> pd.DataFrame:
    
    """
    Creates and saves tile information in tabular format 

        Parameters:
            lst (list): A list of dictionaries containing tile information
            output_folder (str): Path to folder where the csv will be saved
            

        Returns:
            df (pd.DataFrame): a DataFrame with the information of each tile 
    """
    
    now = datetime.now().strftime("%d%m%YT%H%M%S")

    df = pd.DataFrame(lst).fillna(0)
    csv_file = f'{output_folder}/tile_info_{now}.csv'
    
    df.to_csv(csv_file, index=False)

    return df


if __name__ == "__main__":

    """
    Runs loading and tiling of image pairs

    The following files must exist in the working directory: 

    data_path.config: 
        Specifies the path of the local folder containing input SAR and ice chart images
        This needs to be configured by each user and will vary from system to system
    
    tile.config:
        Specifies the output path for the tiled sar images and tiled ice charts
        This needs to be configured by each user and will vary from system to system

    constants.py: 
        Contains a list of the image pairs to be processed
        as well as associated metadata such as geographical region

    """
    # Parse command line arguments
    parser = ArgumentParser(description="Sea Ice Tiling")
    parser.add_argument("--mode", default="train/val", type=str, choices=["train/val", "test"],
                        help="Whether to tile train/val images or test images")
    parser.add_argument("--n_pairs", default=1, type=int, help="Number of pairs to process")
    parser.add_argument("--resolution", default=256, type=int, help="Width and height of resulting tiles")
    parser.add_argument("--stride", default=None, type=int, help="Stride of the sliding window")
    parser.add_argument("--flip_charts", default='True', type=str, help="Whether to flip charts vertically at writing")
    parser.add_argument("--sar_band3", default='True', type=str, help="Whether to save a separate file with HH/HV as band 3")
    parser.add_argument("--sar_folder", default='sar', type=str, help="SAR output folder name")
    parser.add_argument("--sar_band3_folder", default='sar_band3', type=str, help="SAR band3 output folder name")
    parser.add_argument("--chart_folder", default='chart', type=str, help="Ice Chart output folder name")
    parser.add_argument("--nan_threshold", default=0.0, type=float, help="Nan threshold to filter tiles")
    args = parser.parse_args()

    # User config
    n_pairs_to_process = args.n_pairs
    resolution = args.resolution
    stride = args.stride
    flip_charts = args.flip_charts  # ice charts may need vertical flip before tiling
    chart_ext = "tiff"
    sar_ext = "tif"
    if args.mode == "test":
        from constants import test_chart_sar_pairs as chart_sar_pairs
        output_folder = Path(open("tile.config").read().strip()) / "test"
        ftp_folder = Path(open("ftp.config").read().strip()) / "test_ims"
        chart_folder = ftp_folder / "rasterised_ice_charts"
        sar_folder = ftp_folder / "original_sar_images"
    else:
        from constants import chart_sar_pairs
        output_folder = Path(open("tile.config").read().strip())
        ftp_folder = Path(open("ftp.config").read().strip())
        chart_folder = ftp_folder / "rasterised_shapefiles"
        sar_folder = ftp_folder / "dual_band_images"
    
    # Prepare to run
    t_start = default_timer()
    total_img = 0
    total_discarded = 0
    total_info = []

    # Run loading and tiling of image pairs
    for i, (chart_name, sar_name, region) in enumerate(chart_sar_pairs):
        if i >= n_pairs_to_process:
            break
        chart_image = load_raster(str(Path(f"{chart_folder}/{chart_name}.{chart_ext}")), default_name="Ice Chart")
        if flip_charts == 'True':
            chart_image = chart_image.reindex(y=chart_image.y[::-1])  # flip vertically
        sar_image = load_raster(str(Path(f"{sar_folder}/{sar_name}.{sar_ext}")), default_name="SAR Image")
        name_extract = re.findall("H_[0-9]{8}T[0-9]{6}", sar_name)[0][2:10]  # use sar datetime as identifier for all outputs
        print(f"Tiling {name_extract} ...")
        img_n, discarded_tiles, info_lst = tile_raster(sar_image, chart_image, output_folder, name_extract, region, nan_threshold=args.nan_threshold,
                                                       size_x=resolution, size_y=resolution, stride_x=stride, stride_y=stride, sar_band3=args.sar_band3,
                                                       sar_subfolder=args.sar_folder, sar_band3_subfolder=args.sar_band3_folder, chart_subfolder=args.chart_folder)
        total_img += img_n; total_discarded += discarded_tiles
        total_info.extend(info_lst)

    create_tile_info_dataframe(total_info, output_folder)

    print("TILING COMPLETE\n")
    print(f"Total image pairs generated: {total_img}")
    print(f"Total discarded tile pairs: {total_discarded}")
    print(f"Proportion discarded: {total_discarded/(total_img+total_discarded)}")

    t_end = default_timer()
    print(f"Execution time: {(t_end - t_start)/60.0} minutes for {n_pairs_to_process} pair(s) of source image(s)")
