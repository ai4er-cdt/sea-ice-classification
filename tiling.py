
import xarray  as xr
import rioxarray as rxr
from xarray.core.dataarray import DataArray


def load_raster(file_path: str, parse_coordinates: bool=False, default_name: str=None, crs: int=4326) -> DataArray:

    '''
    Loads and returns an xarray.core.dataarray.DataArray

            Parameters:
                    file_path (str): Path to file
                    parse_coordinates (bool): Parses the coordinates of the file, if any
                    default_name (str): Name for the array
                    crs (int): CRS of the resulting array

            Returns:
                    raster (xarray.core.dataarray.DataArray): DataArray from file
    '''

    raster = rxr.open_rasterio(file_path, parse_coordinates=parse_coordinates, default_name=default_name)

    assert type(raster) == xr.core.dataarray.DataArray # Makes sure the data structure is DataArray
    raster.rio.write_crs(crs, inplace=True) # Modifies CRS of raster files

    return raster

def check_null_values(dataArray: DataArray, nan_threshold: float, size_x: int, size_y: int) -> bool:

    '''
    Loads and returns an xarray.core.dataarray.DataArray

            Parameters:
                    dataArray (xarray.core.dataarray.DataArray): DataArray object from the xarray library
                    nan_threshold (float): number in [0,1]
                    size_x (int): Size of the tile in the horizontal axis
                    size_y (int): Size of the tile in the vertical axis

            Returns:
                    ans (bool): Whether the number of Nans is greater than the input threshold
    '''

    assert 0.0 <= nan_threshold <= 1.0  # Makes sure the parameter falls within the desired range

    sub_null = dataArray.isnull() # Checks null values in array
    groups = sub_null.groupby(sub_null).groups # Makes a dictionary of positions of Nan and non-Nan values
    na_count = {key: len(groups[key]) for key in groups.keys()} # Counts Nan and non-Nan values

    # Checks if the number of Nans in the raster is less than the threshold, if any
    ans = (True in groups.keys() and na_count[True] / (size_x * size_y) < nan_threshold) or True not in groups.keys()

    return ans

def tile_raster(sar_image: DataArray, ice_chart: DataArray, data_folder: str, basename: str,
                size_x: int=256, size_y: int=256, start_x: int=0, start_y: int=0,
                end_x: int=None, end_y: int=None, stride_x: int=128, stride_y: int=128,
                nan_threshold: float=1.0) -> int:

    '''
    Loads and returns an xarray.core.dataarray.DataArray

            Parameters:
                    sar_image (xarray.core.dataarray.DataArray): DataArray object from the xarray library
                    ice_chart (xarray.core.dataarray.DataArray): DataArray object from the xarray library
                    data_folder (str): Path to folder where images will be saved
                    basename (str): Base name for saved files
                    size_x (int): Size of the tile in the horizontal axis
                    size_y (int): Size of the tile in the vertical axis
                    start_x (int): Starting coordinate for tiling in the horizontal axis
                    start_y (int): Starting coordinate for tiling in the vertical axis
                    end_x (int): Final coordinate for tiling in the horizontal axis
                    end_y (int): Final coordinate for tiling in the vertical axis
                    stride_x (int): Stride of the moving window in the horizontal axis
                    stride_y (int): Stride of the moving window in the vertical axis
                    nan_threshold (float): number in [0,1]

            Returns:
                    img_n (int): Number of images generated
    '''

    # Makes sure some parameters have default values according to input
    end_x = sar_image.shape[2] if end_x is None else end_x
    end_y = sar_image.shape[1] if end_y is None else end_y
    stride_x = size_x if stride_x is None else stride_x
    stride_y = size_y if stride_y is None else stride_y

    # Checkpoint of parameters
    assert 0 < size_x <= sar_image.shape[2] // 2
    assert 0 < size_y <= sar_image.shape[1] // 2
    assert 0 <= start_x <= sar_image.shape[2]
    assert 0 <= start_y <= sar_image.shape[1]
    assert 0 < end_x <= sar_image.shape[2]
    assert 0 < end_y <= sar_image.shape[1]
    assert 0 < stride_x <= size_x
    assert 0 < stride_y <= size_y

    # Counter for naming convention
    img_n = 1
    # Counter for discarded tiles
    discarded_tiles = 0

    # Iterates over rows and columns of the image according to input parameters
    for row in range(start_y, end_y, stride_y):
        for col in range(start_x, end_x, stride_x):

            # Tile naming convention
            # Adding a prefix to distinguish data from the Weddell Sea and data from the Antarctic Peninsula 
            if basename in ['20181220', '20181210', '20181209', '20181203']:
                prefix = 'AP'
            else:
                prefix = 'WS'

            file_n = '{:0>5}'.format(img_n)
            filename = f'{prefix}_{basename}_{file_n}_[{col},{row}]_{size_x}x{size_y}.tiff'

            # Indexes ice chart and SAR images according to parameters
            sub_sar = sar_image[:, row:row + size_y, col:col + size_x]
            sub_chart = ice_chart[:, row:row + size_y, col:col + size_x]

            # Checks if the current tile has the same shape as the parameters, if not it continues
            if (sub_sar.shape[1] * sub_sar.shape[2] != size_x * size_y) or (sub_chart.shape[1] * sub_chart.shape[2] != size_x * size_y):

                continue

            # Checks Nan values in the current tile for both the ice chart and the SAR image. 
            # If both are True, It will save the tiles in two different folders
            if not check_null_values(sub_sar, nan_threshold, size_x, size_y) or not check_null_values(sub_chart, nan_threshold, size_x, size_y):
                discarded_tiles += 1
                continue

            # Save files to different folders
            sub_sar.rio.to_raster(data_folder + f'features/ftrs_{filename}')
            sub_chart.rio.to_raster(data_folder + f'labels/lbls_{filename}')

            # Increase counter for naming convention
            img_n += 1

    print (f"Number of discarded tiles: {discarded_tiles}")
    return img_n


# # EXAMPLE
# import re
# from constants import chart_sar_pairs

# ex_chart_image = load_raster('FTP_data/rasterised_shapefiles/' + chart_sar_pairs[0][0] + '.tiff', default_name='Ice Chart')
# ex_sar_image = load_raster('FTP_data/dual_band_images/' + chart_sar_pairs[0][1] + '.tif', default_name='SAR Image')
# name_extract = re.findall('H_[0-9]{8}T', chart_sar_pairs[0][1])
# new_title = re.sub('H_', '', name_extract[0])
# tile_raster(ex_sar_image, ex_chart_image, 'tiled_images/', new_title, size_x=256, size_y=256)
