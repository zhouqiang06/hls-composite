"""Extract a values from an HLS time series for a set of points in a MGRS tile"""

import argparse
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
from maap.maap import MAAP
from pystac import Asset, Catalog, CatalogType, Item
from rasterio.session import AWSSession
from rustac import DuckdbClient

import os
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime
# from glob import glob

import geopandas
from osgeo import gdal
import rasterio as rio
import rioxarray as rxr
import earthaccess

import dask.array as da
# from dask.diagnostics import ProgressBar


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logging.getLogger("botocore").setLevel(logging.WARNING)
logger = logging.getLogger("HLSComposite")

# GDAL configurations used to successfully access LP DAAC Cloud Assets via vsicurl
gdal.SetConfigOption('GDAL_HTTP_COOKIEFILE','~/cookies.txt')
gdal.SetConfigOption('GDAL_HTTP_COOKIEJAR', '~/cookies.txt')
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN','EMPTY_DIR')
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS','TIF')
gdal.SetConfigOption('GDAL_HTTP_UNSAFESSL', 'YES')

GDAL_CONFIG = {
    "CPL_TMPDIR": "/tmp",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "TIF,GPKG",
    "GDAL_CACHEMAX": "512",
    "GDAL_INGESTED_BYTES_AT_OPEN": "32768",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "PYTHONWARNINGS": "ignore",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "536870912",
    "GDAL_NUM_THREADS": "ALL_CPUS",
    # "CPL_DEBUG": "ON" if debug else "OFF",
    # "CPL_CURL_VERBOSE": "YES" if debug else "NO",
}

# LPCLOUD S3 CREDENTIAL REFRESH
CREDENTIAL_REFRESH_SECONDS = 50 * 60


class CredentialManager:
    """Thread-safe credential manager for S3 access"""

    def __init__(self):
        self._lock = threading.Lock()
        self._credentials: dict[str, Any] | None = None
        self._fetch_time: float | None = None
        self._session: AWSSession | None = None

    def get_session(self) -> AWSSession:
        """Get current session, refreshing credentials if needed"""
        with self._lock:
            now = time.time()

            # Check if credentials need refresh
            if (
                self._credentials is None
                or self._fetch_time is None
                or (now - self._fetch_time) > CREDENTIAL_REFRESH_SECONDS
            ):
                logger.info("fetching/refreshing S3 credentials")
                self._credentials = self._fetch_credentials()
                self._fetch_time = now
                self._session = AWSSession(**self._credentials)

            return self._session

    @staticmethod
    def _fetch_credentials() -> dict[str, Any]:
        """Fetch new credentials from MAAP"""
        maap = MAAP(maap_host="api.maap-project.org")
        creds = maap.aws.earthdata_s3_credentials(
            "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials"
        )
        return {
            "aws_access_key_id": creds["accessKeyId"],
            "aws_secret_access_key": creds["secretAccessKey"],
            "aws_session_token": creds["sessionToken"],
            "region_name": "us-west-2",
        }


# Global credential manager instance
_credential_manager = CredentialManager()

HLS_COLLECTIONS = ["HLSL30_2.0", "HLSS30_2.0"]
HLS_STAC_GEOPARQUET_HREF = "s3://nasa-maap-data-store/file-staging/nasa-map/hls-stac-geoparquet-archive/v2/{collection}/**/*.parquet"

URL_PREFIX = "https://data.lpdaac.earthdatacloud.nasa.gov/"
DTYPE = "int16"
FMASK_DTYPE = "uint8"
NODATA = -9999
FMASK_NODATA = 255

BAND_MAPPING = {
    "HLSL30_2.0": {
        "coastal_aerosol": "B01",
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir_narrow": "B05",
        "swir_1": "B06",
        "swir_2": "B07",
        "cirrus": "B09",
        "thermal_infrared_1": "B10",
        "thermal": "B11",
        "Fmask": "Fmask",
    },
    "HLSS30_2.0": {
        "coastal_aerosol": "B01",
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "red_edge_1": "B05",
        "red_edge_2": "B06",
        "red_edge_3": "B07",
        "nir_broad": "B08",
        "nir_narrow": "B8A",
        "water_vapor": "B09",
        "cirrus": "B10",
        "swir_1": "B11",
        "swir_2": "B12",
        "Fmask": "Fmask",
    },
}
# these are the ones that we are going to use
DEFAULT_BANDS = [
    "red",
    "green",
    "blue",
    "nir_narrow",
    "swir_1",
    "swir_2",
    "Fmask",
]
DEFAULT_RESOLUTION = 30



common_bands = ["Blue","Green","Red","NIR_Narrow","SWIR1", "SWIR2", "Fmask"]

L8_bandname = {"B01":"Coastal_Aerosol", "B02":"Blue", "B03":"Green", "B04":"Red", 
               "B05":"NIR_Narrow", "B06":"SWIR 1", "B07":"SWIR 2", "B09":"Cirrus"}
S2_bandname = {"B01":"Coastal_Aerosol", "B02":"Blue", "B03":"Green", "B04":"Red", 
               "B8A":"NIR_Narrow", "B11":"SWIR1", "B12":"SWIR2", "B10":"Cirrus"}

L8_name2index = {'Coastal_Aerosol': 'B01', 'Blue': 'B02', 'Green': 'B03', 'Red': 'B04',
                 'NIR_Narrow': 'B05', 'SWIR1': 'B06', 'SWIR2': 'B07', 'Fmask': 'Fmask'}
S2_name2index = {'Coastal_Aerosol': 'B01', 'Blue': 'B02', 'Green': 'B03', 'Red': 'B04', 
                 'NIR_Edge1': 'B05', 'NIR_Edge2': 'B06', 'NIR_Edge3': 'B07', 
                  'NIR_Broad': 'B08', 'NIR_Narrow': 'B8A', 'SWIR1': 'B11', 'SWIR2': 'B12', 'Fmask': 'Fmask'}

sr_scale = 0.0001
ang_scale = 0.01
SR_FILL = -9999
QA_FILL = 255 #FMASK_FILL

QA_BIT = {'cirrus': 0,
'cloud': 1,
'adj_cloud': 2,
'cloud shadow':3,
'snowice':4,
'water':5,
'aerosol_l': 6,
'aerosol_h': 7
}

chunk_size = dict(band=1, x=1098, y=1098)

def mask_hls(qa_arr, mask_list=['cloud', 'adj_cloud', 'cloud shadow']):
    # This function takes the HLS QA array as input and exports the cloud mask array. 
    # The mask_list assigns the QA conditions you would like to mask.
    msk = np.zeros_like(qa_arr)#.astype(bool)
    for m in mask_list:
        if m in QA_BIT.keys():
            msk += (qa_arr & 1 << QA_BIT[m]) > 0
        if m == 'aerosol_high':
            msk += ((qa_arr & (1 << QA_BIT['aerosol_h'])) > 0) * ((qa_arr & (1 << QA_BIT['aerosol_l'])) > 0)
        if m == 'aerosol_moderate':
            msk += ((qa_arr & (1 << QA_BIT['aerosol_h'])) > 0) * ((qa_arr | (1 << QA_BIT['aerosol_l'])) != qa_arr)
        if m == 'aerosol_low':
            msk += ((qa_arr | (1 << QA_BIT['aerosol_h'])) != qa_arr) * ((qa_arr & (1 << QA_BIT['aerosol_l'])) > 0)
    return msk > 0


def get_geo(filepath):
    # ds = gdal.Open(filepath)
    # return ds.GetGeoTransform(), ds.GetProjection()
    with rio.open(filepath) as ds:
        return ds.transform, ds.crs


def saveGeoTiff(filename, data, template_file):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if os.path.exists(filename):
        os.remove(filename)

    if data.ndim == 2:
        nband = 1
    else:
        nband = data.shape[2]
    try:
        with rio.open(template_file) as ds:
            output_transform, output_crs = ds.transform, ds.crs
            profile = {
                        'driver': 'GTiff',
                        'dtype': data.dtype,
                        'count': nband,  # Number of bands
                        'height': data.shape[0],
                        'width': data.shape[1],
                        'crs': output_crs,
                        'transform': output_transform,
                        'compress': 'lzw' # Optional: add compression
                    }
        with rio.open(filename, 'w', **profile) as dst:
            if nband == 1:
                dst.write(data, 1) # Write the array to band 1
            else:
                dst.write(data)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")


def load_band_retry(tif_path: Path, max_retries: int = 3, delay: int = 5, fill_value=SR_FILL) -> np.ma.masked_array:
    for attempt in range(max_retries):
        try:
            return rxr.open_rasterio(tif_path, lock=False, chunks=chunk_size, masked=True, fill_value=fill_value).squeeze()
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {tif_path}: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    raise RuntimeError(f"Failed to read {tif_path} after {max_retries} attempts")


def read_sr_band(tif_path: Path) -> np.ma.masked_array:
    data = load_band_retry(tif_path, fill_value=SR_FILL)
    return np.ma.masked_less_equal(data, 0)


def apply_fmask(data: np.ndarray, fmask: np.ndarray) -> np.ma.masked_array:
    return np.ma.masked_array(data, fmask)


def apply_union_of_masks(bands: list[np.ma.masked_array]) -> list[np.ma.masked_array]:
    mask = np.ma.nomask
    for band in bands:
        mask = np.ma.mask_or(mask, band.mask)

    for band in bands:
        band.mask = mask

    return bands


def get_meta(file_path: str):
    return rio.open(file_path).meta


def find_tile_bounds(tile: str):
    # gdf = geopandas.read_file(r"s3://maap-ops-workspace/shared/zhouqiang06/AuxData/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp")
    gdf = geopandas.read_file(r"/projects/my-public-bucket/AuxData/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp")
    bounds_list = [np.round(c, 3) for c in gdf[gdf["Name"]==tile].bounds.values[0]]
    return tuple(bounds_list)


def get_HLS_data(tile:str, bandnum:int, start_date:str, end_date:str):
    # from rustac import DuckdbClient
    client = DuckdbClient(use_hive_partitioning=True)
    # configure duckdb to find S3 credentials
    client.execute(
        """
        CREATE OR REPLACE SECRET secret (
            TYPE S3,
            PROVIDER CREDENTIAL_CHAIN
        );
        """
    )
    response = client.search(
        href="s3://maap-ops-workspace/shared/henrydevseed/hls-stac-geoparquet-v1/year=*/month=*/*.parquet",
        datetime=f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        bbox=find_tile_bounds(tile),
    )
    results = GetBandLists_HLS_STAC(response, tile, bandnum)
    return results


def GetBandLists_HLS_STAC(response, tile:str, bandnum:int):
    BandList = []
    for i in range(len(response)):
        product_type = response[i]['id'].split('.')[1]
        if product_type=='L30':
            bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B05', 6:'B06', 7:'B07',8:'Fmask'})
        elif product_type=='S30':
            bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B8A', 6:'B11', 7:'B12',8:'Fmask'})
        else:
            print("HLS product type not recognized: Must be L30 or S30.")
            os._exit(1)
            
        try:
            getBand = response[i]['assets'][bands[bandnum]]['href']
            if filter_url(getBand, tile, bands[bandnum]):
                BandList.append(getBand)
        except Exception as e:
            print(e)
    return BandList


def filter_url(url: str, tile: str, band: str):
    if (os.path.basename(url).split('.')[2][1:]==tile) & (url.endswith(f"{band}.tif")):
        return True
    return False    


def get_tile_urls(tile:str, bandnum:int, start_date:str, end_date:str, access_type="external"):
    url_list = []
    try:
        results = earthaccess.search_data(short_name=f"HLSL30",
                                        cloud_hosted=True,
                                        temporal = (start_date, end_date), #"2022-07-17","2022-07-31"
                                        bounding_box = find_tile_bounds(tile), #bounding_box = (-51.96423,68.10554,-48.71969,70.70529)
                                        )
    except:
        results = []
    if len(results) > 0:
        bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B05', 6:'B06', 7:'B07',8:'Fmask'})
        for rec in results:
            for url in rec.data_links(access=access_type):
                if filter_url(url, tile, bands[bandnum]):
                    url_list.append(url)
    try:
        results = earthaccess.search_data(short_name=f"HLSS30",
                                        cloud_hosted=True,
                                        temporal = (start_date, end_date), #"2022-07-17","2022-07-31"
                                        bounding_box = find_tile_bounds(tile), #bounding_box = (-51.96423,68.10554,-48.71969,70.70529)
                                        )
    except:
        results = []
    if len(results) > 0:
        bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B8A', 6:'B11', 7:'B12',8:'Fmask'})
        for rec in results:
            for url in rec.data_links(access=access_type):
                if filter_url(url, tile, bands[bandnum]):
                    url_list.append(url)
    return url_list


def find_all_granules(tile: str, bandnum: int, start_date: str, end_date: str, search_source="STAC"):
    if search_source.lower() == "stac":
        url_list = get_HLS_data(tile=tile, bandnum=bandnum, start_date=start_date, end_date=end_date)
    elif search_source.lower() == "earthaccess":
        url_list = get_tile_urls(tile=tile, bandnum=bandnum, start_date=start_date, end_date=end_date, access_type="external")
    else:
        print("search_source not recognized. Must be 'STAC' or 'earthaccess'.")
        # os._exit(1)
    if len(url_list) == 0:
        print("No granules found.")
        return pd.DataFrame()
    sat_list = [os.path.basename(g).split('.')[1] for g in url_list] # sat from HLS.L10.T18SUJ.2020010T155225.v2.0
    date_list = [datetime.strptime(os.path.basename(g).split('.')[3][:7], "%Y%j") for g in url_list] # date from HLS.L10.T18SUJ.2020010T155225.v2.0
    return pd.DataFrame({"Date": date_list, "Sat": sat_list, "granule_path": url_list})


# EVI2 = 2.5 * ((NIR - R) / (NIR + 2.4*R + 1)) derived for MODIS
def evi2(red_arr, nir_arr):
    red_arr = red_arr * sr_scale
    nir_arr = nir_arr * sr_scale
    red_arr[red_arr<=0] = np.nan
    nir_arr[nir_arr<=0] = np.nan
    arr = 2.5 * (nir_arr - red_arr) / (nir_arr + 2.4*red_arr + 1)
    arr[(nir_arr + 2.4*red_arr + 1) == 0] = np.nan
    arr[np.isinf(arr)] = np.nan
    return arr

def ndwi(green_arr, nir_arr):
    green_arr = green_arr * sr_scale
    nir_arr = nir_arr * sr_scale
    green_arr[green_arr<=0] = np.nan
    nir_arr[nir_arr<=0] = np.nan
    arr = (green_arr - nir_arr) / (green_arr + nir_arr)
    arr[(green_arr + nir_arr) == 0] = np.nan
    arr[np.isinf(arr)] = np.nan
    return arr


def createVIstack(granlue_dir_df: list):
    '''Calculate VI for each source scene
    Mask out pixels above or below the red band reflectance range limit values'''
    # chunk_size = dict(band=1, x=1098, y=1098)
    # idx = 0
    # for index, g_rec in tqdm(granlue_dir_df.iterrows(), total=granlue_dir_df.shape[0]):
    for idx, g_rec in granlue_dir_df.iterrows():
        try:
            fmask = load_band_retry(g_rec.granule_path, fill_value=QA_FILL).astype(np.uint8)
        except:
            fmask = np.zeros((3660, 3660), dtype=np.uint8) + QA_FILL
        fmaskarr_by = mask_hls(fmask, mask_list=['cloud', 'adj_cloud', 'cloud shadow', 'aerosol_h']) | (fmask == QA_FILL)
        water_mask_by = ~fmaskarr_by & mask_hls(fmask, mask_list=['water', 'snowice'])
        if idx == 0:
            # fmask_rasters = da.from_array(fmaskarr_by, chunks=chunk_size)
            fmask_rasters = da.expand_dims(fmaskarr_by, axis=0)
            # water_mask_rasters = da.from_array(water_mask_by, chunks=chunk_size)
            water_mask_rasters = da.expand_dims(water_mask_by, axis=0)
        else:
            fmask_rasters = da.concatenate([fmask_rasters, da.expand_dims(fmaskarr_by, axis=0)], axis=0)
            water_mask_rasters = da.concatenate([water_mask_rasters, da.expand_dims(water_mask_by, axis=0)], axis=0)
        if g_rec.Sat in ["L30", "L10"]:
            band_dict = L8_name2index
        elif g_rec.Sat in ["S30", "S10"]:
            band_dict = S2_name2index
        else:
            print("Sat value wrong.")
        data = apply_union_of_masks(
            [apply_fmask(read_sr_band(g_rec.granule_path.replace("Fmask", band_dict[band])), fmaskarr_by) for band in common_bands]
        )
        evi_arr = evi2(data[common_bands.index("Red")], data[common_bands.index("NIR_Narrow")])
        ndwi_arr = ndwi(data[common_bands.index("Green")], data[common_bands.index("NIR_Narrow")])

        if idx == 0:
            evi_rasters = da.from_array(evi_arr, chunks=chunk_size)
            evi_rasters = da.expand_dims(evi_rasters, axis=0)
            ndwi_rasters = da.from_array(ndwi_arr, chunks=chunk_size)
            ndwi_rasters = da.expand_dims(ndwi_rasters, axis=0)
        else:
            evi_rasters = da.concatenate([evi_rasters, da.expand_dims(evi_arr, axis=0)], axis=0)
            ndwi_rasters = da.concatenate([ndwi_rasters, da.expand_dims(ndwi_arr, axis=0)], axis=0)
        # idx += 1
        # vi_rasters.append(vi_arr)
    evi_rasters = da.ma.masked_array(evi_rasters, mask=fmask_rasters, chunks=chunk_size, fill_value=SR_FILL).compute()
    water_mask = da.ma.masked_array(water_mask_rasters, mask=fmask_rasters, chunks=chunk_size, fill_value=SR_FILL)
    water_mask = water_mask.all(axis=0).compute()
    if np.any(water_mask):
        print("Combine VI for permanent water")
        row_indices, col_indices = np.argwhere(water_mask).T
        evi_rasters[:, row_indices, col_indices] = ndwi_rasters.compute()[:, row_indices, col_indices] # if it is permanent water, use maximum NDWI
    return evi_rasters#, ndwi_rasters


def print_array_stats(result):
    print(f"\tPrinting array stats: {result.shape}")
    # Count of valid (non-NaN) pixels
    valid_pixel_count = da.count_nonzero(~np.isnan(result))
    print(f"\t\tValid Pixel Count: {valid_pixel_count}")
    print(f"\t\tMean: {da.nanmean(result):.2f}")
    print(f"\t\tStandard Deviation: {np.nanstd(result):.2f}")
    print(f"\t\tMinimum: {da.nanmin(result):.2f}")
    print(f"\t\tMaximum: {da.nanmax(result):.2f}")
    if not da.any(da.isnan(result)):
        # print(np.sum(~np.isnan(result)), result[~np.isnan(result)].flatten().shape())
        print(f"\t\t25th Percentile: {da.nanpercentile(da.ma.filled(result, da.nan), 25):.2f}")
        print(f"\t\t50th Percentile (Median): {da.nanpercentile(da.ma.filled(result, da.nan), 50):.2f}")
        print(f"\t\t75th Percentile: {da.nanpercentile(da.ma.filled(result, da.nan), 75):.2f}")


def safe_nanarg_stat(arr, stat='max', axis=0):
    """
    Applies nanargmax safely to prevent `ValueError` in all-NaN slices.
    
    Parameters:
    - arr (np.array): Numpy array to compute maximum index ignoring NaNs.
    - axis (int): Axis along which to find the maximum index.
    
    Returns:
    - np.array: Indices of the maximum values along the given axis, avoiding all-NaN slices.
    """
    # arr_dask = da.from_array(arr, chunks=chunk_size)
    # Check where rows/columns are all NaNs along a specified axis
    all_nan_mask = da.all(da.isnan(arr), axis=axis)
    # arr[0,:,:][all_nan_mask] = SR_FILL # For all-NaN slices ValueError is raised.
    # Replace all-NaN slices with a fill value (e.g., a large negative number)
    # fill_value = da.min(arr) if da.min(arr) < 0 else -da.inf
    fill_value = SR_FILL
    # Choose an appropriate fill value based on your data context
    arr = da.where(all_nan_mask[np.newaxis, ...], fill_value, arr)
    # arr_dask = da.from_array(arr_filled, chunks=(arr.shape[0], 100, 100))
    if stat == 'max':
        # Return indices, using nanargmax safely
        arr = da.nanargmax(arr, axis=axis) # index_arr
    elif stat == 'min':
        arr = da.nanargmin(arr, axis=axis)
    else:
        raise ValueError("Invalid statistic. Choose from 'min', 'max'.")
    # with ProgressBar():
    #     arr = arr.compute()
    return arr, all_nan_mask


def compute_stat_from_masked_array(masked_array, no_data_value = None, stat='max'):
    """
    Compute the pixel-wise statistic from a numpy masked 3D array, accounting for NaN values and a no data value,
    and return a 2D array.
    
    Parameters:
    - masked_array (np.ma.MaskedArray): A masked 3D array.
    - no_data_value (scalar): The value to mask out from the data before computing statistics.
    - stat (str): The statistic to compute - 'min', 'max', or 'percentile'.
    - percentile_value (float): The percentile value to compute if stat is 'percentile'.
    
    Returns:
    - np.array: A 2D array containing the computed statistic for each pixel.
    """
    
    data = da.ma.filled(masked_array, np.nan)  # Convert masked values to NaN

    if no_data_value is not None:
        print("\tApply the mask for no data values...")
        data = da.ma.masked_array(data, mask=(data == no_data_value), chunks=chunk_size)
        data = da.ma.filled(data, np.nan)  # Convert the new mask to NaN
        # print_array_stats(data)
    
    if stat == 'min':
        #result = np.nanargmin(data, axis=0)
        result, all_nan_mask = safe_nanarg_stat(data, stat='min', axis=0)
    elif stat == 'max':
        #result = np.nanargmax(data, axis=0)
        result, all_nan_mask = safe_nanarg_stat(data, stat='max', axis=0)
    else:
        raise ValueError("Invalid statistic. Choose from 'min', 'median', 'max', 'percentile'.")

    # Print statistical summary of the result array
    print(f"\tStatistical summary of index array for stat={stat}")

    return da.ma.masked_array(result, mask=all_nan_mask, fill_value=no_data_value)

def CollapseBands(inArr, NDVItmp, BoolMask):
    '''
    Inserts the bands as arrays (made earlier)
    Creates a single layer by using the binary mask and a sum function to collapse n-dims to 2-dims
    '''
    inArr = da.ma.masked_equal(inArr, 0)
    inArr[da.logical_not(NDVItmp)]=0
    compImg = da.ma.masked_array(inArr.sum(0), BoolMask, chunks=chunk_size)
    # compImg = np.round(compImg / sr_scale, 0)
    return da.ma.filled(compImg, SR_FILL) # compImg.filled(-9999)
    
def CreateComposite(granlue_dir_df: list, band: str, NDVItmp: np.array, BoolMask: np.array):
    #print("\t\tMaskedFile")
    MaskedFile = []
    for g_rec in granlue_dir_df.itertuples():
        if g_rec.Sat in ["L10", "L30"]:
            band_dict = L8_name2index
        elif g_rec.Sat in ["S10", "S30"]:
            band_dict = S2_name2index
        else:
            print("Sat value wrong.")
        if band in band_dict.keys():
            MaskedFile.append(read_sr_band(g_rec.granule_path.replace("Fmask", band_dict[band])))
        else:
            # print("NDVItmp shape: ", NDVItmp.shape, "BoolMask shape: ", BoolMask.shape)
            MaskedFile.append(np.ma.array(np.full((NDVItmp.shape[1], NDVItmp.shape[2]), SR_FILL), mask=True))
    #print("\t\tComposite")
    Composite = CollapseBands(MaskedFile, NDVItmp, BoolMask)
    return Composite

def run(tile: str, start_date: str, end_date: str, stat: str, save_dir: str, search_source="STAC"):
    start_date_doy = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_doy = datetime.strptime(end_date, "%Y-%m-%d")
    granule_df_range = find_all_granules(tile=tile, bandnum=8, start_date=start_date, end_date=end_date, search_source=search_source) # band 8 is Fmask
    print(len(granule_df_range), " granules in date range.")
    if len(granule_df_range) == 0:
        print(f"No granule found from {start_date_doy.strftime("%Y%j")} to {end_date_doy.strftime("%Y%j")}.")
        return

    print(f"Creating VI array.")
    VIstack_ma = createVIstack(granlue_dir_df=granule_df_range)
    # VIstack_ma = da.ma.masked_array(VIstack, chunks=chunk_size)
    print(f"Calculating {stat} VI index.")
    VIstat = compute_stat_from_masked_array(VIstack_ma, no_data_value=SR_FILL, stat=stat)
    BoolMask = da.ma.getmaskarray(VIstat)
    # create a tmp array (binary mask) of the same input shape
    VItmp = da.ma.masked_array(da.zeros(VIstack_ma.shape, dtype=bool, chunks=chunk_size))

    tmp_g_path = granule_df_range.iloc[1]["granule_path"]
    # for each dimension assign the index position (flattens the array to a LUT)
    print(f"Create LUT of VI positions using stat={stat}")
    for i in range(np.shape(VIstack_ma)[0]):
        VItmp[i,:,:]=VIstat==i
    for band in common_bands: #common_bands
        arr = CreateComposite(granule_df_range, band, VItmp, BoolMask)
        if band == "Fmask":
            arr[arr == -9999] = 255
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.int16)
        out_dir = os.path.join(save_dir, tile, f"HLS.M30.T{tile}.{start_date_doy.strftime("%Y%j")}.{end_date_doy.strftime("%Y%j")}.2.0")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = os.path.join(out_dir, f"HLS.M30.T{tile}.{start_date_doy.strftime("%Y%j")}.{end_date_doy.strftime("%Y%j")}.2.0.{band}.tif")
        print(f"Saving {band} band.")
        arr = arr.compute()# / sr_scale
        saveGeoTiff(out_file, arr, template_file=tmp_g_path)

    # Get the pixelwise count of the valid data
    CountComp = da.sum((VIstack_ma != SR_FILL), axis=0) # -9999
    out_file = os.path.join(out_dir, f"HLS.M30.T{tile}.{start_date_doy.strftime("%Y%j")}.{end_date_doy.strftime("%Y%j")}.2.0.ValidCount.tif")
    print(f"Saving count of the valid data.")
    CountComp = CountComp.compute()
    # print(f"Count array min ({CountComp.min()}), max ({CountComp.max()}), and shape ({CountComp.shape})")
    saveGeoTiff(out_file, CountComp, template_file=tmp_g_path)



if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Queries the HLS STAC geoparquet archive and create composite images"
    )
    parse.add_argument(
        "--tile",
        help="MGRS tile id, e.g. 15XYZ",
        required=True,
        type=str,
    )
    parse.add_argument(
        "--start_date",
        help="start date in ISO format (e.g., 2024-01-01)",
        required=True,
        type=str,
    )
    parse.add_argument(
        "--end_date",
        help="end date in ISO format (e.g., 2024-12-31)",
        required=True,
        type=str,
    )
    parse.add_argument(
        "--output_dir", help="Directory in which to save output", required=True
    )
    parse.add_argument(
        "--search_source",
        help="Either STAC or earthaccess to search for HLS granules",
        action="store_true",
        default="STAC",
    )
    args = parse.parse_args()

    output_dir = Path(args.output_dir)

    logger.info(
        f"setting GDAL config environment variables:\n{json.dumps(GDAL_CONFIG, indent=2)}"
    )
    os.environ.update(GDAL_CONFIG)

    logger.info(
        f"running with mgrs_tile: {args.tile}, start_datetime: {args.start_date}, end_datetime: {args.end_date}"
    )

    run(
        tile=args.tile,
        start_date=args.start_date,
        end_date=args.end_date,
        stat="max",
        save_dir=output_dir,
        search_source="STAC",
    )
