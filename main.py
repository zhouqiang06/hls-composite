#!/panfs/ccds02/nobackup/people/qzhou2/miniforge3/envs/hls_mamba/bin/python
# import platform
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime
# from glob import glob

import argparse
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import geopandas
# from osgeo import gdal
import rasterio as rio
import rioxarray as rxr
import earthaccess

import dask.array as da
# from dask.diagnostics import ProgressBar
# from tqdm import tqdm

from maap.maap import MAAP
from pystac import Asset, Catalog, CatalogType, Item
from rasterio.session import AWSSession
from rustac import DuckdbClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logging.getLogger("botocore").setLevel(logging.WARNING)
logger = logging.getLogger("HLSComposite")

GDAL_CONFIG = {
    "CPL_TMPDIR": "/tmp",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "TIF,GPKG,SHP,SHX,PRJ,DBF,JSON,GEOJSON",
    # "GDAL_CACHEMAX": "512",
    # "GDAL_INGESTED_BYTES_AT_OPEN": "32768",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "PYTHONWARNINGS": "ignore",
    # "VSI_CACHE": "TRUE",
    # "VSI_CACHE_SIZE": "536870912",
    "GDAL_NUM_THREADS": "ALL_CPUS",
    "GDAL_HTTP_COOKIEFILE": str(Path.home() / "cookies.txt"),
    "GDAL_HTTP_COOKIEJAR": str(Path.home() / "cookies.txt"),
    "GDAL_HTTP_UNSAFESSL": "YES",
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

sr_scale = 0.0001
ang_scale = 0.01
SR_FILL = -9999
QA_FILL = 255 #FMASK_FILL

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


QA_BIT = {'cirrus': 0,
'cloud': 1,
'adj_cloud': 2,
'cloud shadow':3,
'snowice':4,
'water':5,
'aerosol_l': 6,
'aerosol_h': 7
}

chunk_size = dict(band=1, x=1830, y=1830)
image_size = (3660, 3660)

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


def saveGeoTiff(filename, data, template_file, access_type="external"):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if os.path.exists(filename):
        os.remove(filename)

    if data.ndim == 2:
        nband, height_data, width_data = 1, data.shape[0], data.shape[1]
    else:
        nband, height_data, width_data = data.shape[0], data.shape[1], data.shape[2]
    try:
        # Get session from credential manager if using direct bucket access
        rasterio_env = {}
        if access_type == "direct":
            rasterio_env["session"] = _credential_manager.get_session()
        with rio.Env(**rasterio_env):
            with rio.open(template_file) as ds:
                output_transform, output_crs = ds.transform, ds.crs
                profile = {
                            'driver': 'GTiff',
                            'dtype': data.dtype,
                            'count': nband,  # Number of bands
                            'height': height_data,
                            'width': width_data,
                            'crs': output_crs,
                            'transform': output_transform,
                            'compress': 'lzw' # Optional: add compression
                        }
            with rio.open(filename, 'w', **profile) as dst:
                if nband == 1:
                    dst.write(data, 1)
                else:
                    for i in range(nband):
                        dst.write(data[i], i + 1)
            return True
    except Exception as e:
        print(f"An error occurred: {e}")


def get_stac_items(
    mgrs_tile: str, start_datetime: datetime, end_datetime: datetime
) -> list[Item]:
    logger.info("querying HLS archive")
    client = DuckdbClient(use_hive_partitioning=True)
    client.execute(
        """
        CREATE OR REPLACE SECRET secret (
             TYPE S3,
             PROVIDER CREDENTIAL_CHAIN
        );
        """
    )

    items = []
    for collection in HLS_COLLECTIONS:
        items.extend(
            client.search(
                href=HLS_STAC_GEOPARQUET_HREF.format(collection=collection),
                datetime="/".join(
                    dt.isoformat() for dt in [start_datetime, end_datetime]
                ),
                filter={
                    "op": "and",
                    "args": [
                        {
                            "op": "like",
                            "args": [{"property": "id"}, f"%.T{mgrs_tile}.%"],
                        },
                        {
                            "op": "between",
                            "args": [
                                {"property": "year"},
                                start_datetime.year,
                                end_datetime.year,
                            ],
                        },
                    ],
                },
            )
        )

    logger.info(f"found {len(items)} items")

    return [Item.from_dict(item) for item in items]


def fetch_single_asset(
    asset_href: str,
    fill_value=SR_FILL,
    direct_bucket_access: bool = False,
):
    """
    Fetch data from a single asset.
    """
    try:
        # Get session from credential manager if using direct bucket access
        rasterio_env = {}
        if direct_bucket_access:
            rasterio_env["session"] = _credential_manager.get_session()

        with rio.Env(**rasterio_env):
            # return rxr.open_rasterio(asset_href, lock=False, chunks=chunk_size, driver='GTiff').squeeze()
            with rio.open(asset_href) as src:
                return da.from_array(src.read(1), chunks=chunk_size)


    except Exception as e:
        logger.warning(f"Failed to read {asset_href}: {e}")
        return None # np.full((image_size[0], image_size[1]), fill_value)


def fetch_with_retry(asset_href: Path, max_retries: int = 3, delay: int = 3, fill_value=SR_FILL, access_type="external"):
    for attempt in range(max_retries):
        try:
            return fetch_single_asset(
                asset_href=asset_href,
                fill_value=fill_value,
                direct_bucket_access=(access_type == "direct"),
            )
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = delay #* (2**attempt)
                logger.warning(
                    f"Link {asset_href} attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"All {max_retries} attempts failed for {asset_href}. Last error: {e}"
                )
                return None # np.full((image_size[0], image_size[1]), fill_value)


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
    gdf = geopandas.read_file(r"s3://maap-ops-workspace/shared/zhouqiang06/AuxData/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp")
    # gdf = geopandas.read_file(r"/projects/my-public-bucket/AuxData/Sentinel-2-Shapefile-Index-master/sentinel_2_index_shapefile.shp")
    bounds_list = [np.round(c, 3) for c in gdf[gdf["Name"]==tile].bounds.values[0]]
    return tuple(bounds_list)


def get_HLS_data(tile:str, bandnum:int, start_date:str, end_date:str, access_type="external"):
    print("Searching HLS STAC Geoparquet archive for HLS data...")
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
    results = []
    for collection in HLS_COLLECTIONS:
        response = client.search(
            href=HLS_STAC_GEOPARQUET_HREF.format(collection=collection),
            datetime=f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
            bbox=find_tile_bounds(tile),
            )
        results.extend(
            GetBandLists_HLS_STAC(response, tile, bandnum)
            )
    if access_type=="direct":
        results = [r.replace(URL_PREFIX, "s3://") for r in results]
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
    print("Searching EarthAccess for HLS data...")
    url_list = []
    try:
        results = earthaccess.search_data(short_name=f"HLSL30",
                                        cloud_hosted=True,
                                        temporal = (start_date, end_date), #"2022-07-17","2022-07-31"
                                        bounding_box = find_tile_bounds(tile), #bounding_box = (-51.96423,68.10554,-48.71969,70.70529)
                                        )
    except Exception as e:
        print(f"An error occurred searching HLSL30: {e}")
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
    except Exception as e:
        print(f"An error occurred searching HLSS30: {e}")
        results = []
    if len(results) > 0:
        bands = dict({2:'B02', 3:'B03', 4:'B04', 5:'B8A', 6:'B11', 7:'B12',8:'Fmask'})
        for rec in results:
            for url in rec.data_links(access=access_type):
                if filter_url(url, tile, bands[bandnum]):
                    url_list.append(url)
    return url_list


def find_all_granules(tile: str, bandnum: int, start_date: str, end_date: str, search_source="STAC", access_type="external"):
    if search_source.lower() == "stac":
        url_list = get_HLS_data(tile=tile, bandnum=bandnum, start_date=start_date, end_date=end_date, access_type=access_type)
    elif search_source.lower() == "earthaccess":
        url_list = get_tile_urls(tile=tile, bandnum=bandnum, start_date=start_date, end_date=end_date, access_type=access_type) 
    else:
        print("search_source not recognized. Must be 'STAC' or 'earthaccess'.")
        # os._exit(1)
    if len(url_list) == 0:
        print("No granules found.")
        return pd.DataFrame()
    sat_list = [os.path.basename(g).split('.')[1] for g in url_list] # sat from HLS.L10.T18SUJ.2020010T155225.v2.0
    date_list = [datetime.strptime(os.path.basename(g).split('.')[3][:7], "%Y%j") for g in url_list] # date from HLS.L10.T18SUJ.2020010T155225.v2.0
    return pd.DataFrame({"Date": date_list, "Sat": sat_list, "granule_path": url_list})



def run(tile: str, start_date, end_date, stat: str, save_dir: str, search_source="STAC", access_type="direct", percentile_value=50):
    start_date_doy = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_doy = datetime.strptime(end_date, "%Y-%m-%d")
    
    # 1. Search for granules
    granule_df = find_all_granules(tile=tile, bandnum=8, start_date=start_date, end_date=end_date, 
                                   search_source=search_source, access_type=access_type)
    
    if len(granule_df) == 0:
        logger.warning(f"No granules found for {tile}")
        return

    out_dir = os.path.join(save_dir, tile, start_date[:4], 
                           f"HLS.M30.T{tile}.{start_date_doy.strftime('%Y%j')}.{end_date_doy.strftime('%Y%j')}.2.0")
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Build Lazy 3D Stacks
    band_stack_dict = {}
    for band in common_bands:
        urls = []
        for row in granule_df.itertuples():
            mapping = L8_name2index if row.Sat in ["L30", "L10"] else S2_name2index
            urls.append(row.granule_path.replace("Fmask", mapping[band]))
        
        band_stack_dict[band] = da.stack([
            fetch_with_retry(u, fill_value=(QA_FILL if band == "Fmask" else SR_FILL), access_type=access_type)
            for u in urls
        ], axis=0)

    # 3. Masking and "Has Data" Identification
    fmask_stack = band_stack_dict["Fmask"]
    bad_pixel_mask = (
        ((fmask_stack & (1 << QA_BIT['cloud'])) > 0) | 
        ((fmask_stack & (1 << QA_BIT['adj_cloud'])) > 0) | 
        ((fmask_stack & (1 << QA_BIT['cloud shadow'])) > 0) |
        (((fmask_stack & (1 << QA_BIT['aerosol_h'])) > 0) & ((fmask_stack & (1 << QA_BIT['aerosol_l'])) > 0)) |
        (fmask_stack == QA_FILL)
    )
    
    # Pre-calculate where we have NO data across time
    all_nan_mask = da.all(bad_pixel_mask, axis=0)
    # Add a singleton dimension for broadcasting to 3D stacks
    all_nan_mask_3d = all_nan_mask[None, :, :]

    # 4. Safe Index Selection (EVI2)
    red = band_stack_dict["Red"].astype(np.float32) * sr_scale
    nir = band_stack_dict["NIR_Narrow"].astype(np.float32) * sr_scale
    evi2_stack = 2.5 * (nir - red) / (nir + 2.4 * red + 1)
    
    # Replace bad pixels with NaNs for stat calculation, 
    # then fill All-NaN slices with a dummy 0 so reductions don't crash
    calc_stack = da.where(bad_pixel_mask, np.nan, evi2_stack)
    safe_calc_stack = da.where(all_nan_mask_3d, 0.0, calc_stack)

    if stat == 'max':
        best_idx = da.nanargmax(safe_calc_stack, axis=0)
    elif stat == 'min':
        best_idx = da.nanargmin(safe_calc_stack, axis=0)
    else:
        # Quantile is particularly sensitive to all-NaN blocks
        target_val = da.nanquantile(safe_calc_stack, percentile_value / 100.0, axis=0)
        diff = da.abs(safe_calc_stack - target_val)
        best_idx = da.nanargmin(diff, axis=0)
    
    # Force index to 0 for all-NaN areas (this pixel will be masked out anyway)
    best_idx = da.where(all_nan_mask, 0, best_idx)
    template_path = granule_df.iloc[0]["granule_path"]

    # 5. Materialize Bands and Handle Std Dev
    for band in common_bands:
        logger.info(f"Processing band: {band}")
        
        current_stack = band_stack_dict[band]
        fill = QA_FILL if band == "Fmask" else SR_FILL
        
        # Best composite value
        comp_band_lazy = da.choose(best_idx, current_stack)
        comp_result = da.where(all_nan_mask, fill, comp_band_lazy)

        if band != "Fmask":
            # Safety for nanstd: replace bad with NaN, then All-NaN slices with dummy 0
            masked_data = da.where(bad_pixel_mask, np.nan, current_stack.astype(np.float32))
            safe_std_stack = da.where(all_nan_mask_3d, 0.0, masked_data)
            
            # If only one image, std is 0
            if len(granule_df) > 1:
                std_calc = da.nanstd(safe_std_stack, axis=0)
            else:
                std_calc = da.zeros_like(all_nan_mask, dtype=np.float32)
                
            std_result = da.where(all_nan_mask, 0, std_calc)
            
            # Execute computation
            comp_out, std_out = da.compute(comp_result, std_result)
            
            std_file = os.path.join(out_dir, f"{os.path.basename(out_dir)}.{band}.std.tif")
            saveGeoTiff(std_file, std_out.round().astype(np.uint16), 
                        template_file=template_path, access_type=access_type)
        else:
            comp_out = comp_result.compute()

        out_file = os.path.join(out_dir, f"{os.path.basename(out_dir)}.{band}.tif")
        saveGeoTiff(out_file, comp_out.astype(np.int16 if band != "Fmask" else np.uint8), 
                    template_file=template_path, access_type=access_type)

    # 6. Metadata Layers
    valid_count = da.sum(~bad_pixel_mask, axis=0).astype(np.uint8)
    saveGeoTiff(os.path.join(out_dir, f"{os.path.basename(out_dir)}.ValidCount.tif"), 
                valid_count.compute(), template_file=template_path)

    doy_values = np.array([int(datetime.strptime(os.path.basename(p).split('.')[3][:7], "%Y%j").strftime("%j")) 
                           for p in granule_df.granule_path])
    start_doy = int(start_date_doy.strftime("%j"))
    
    doy_stack = da.from_array(doy_values[:, None, None], chunks=(1, 512, 512))
    best_doy = da.choose(best_idx, doy_stack)
    relative_doy = da.where(all_nan_mask, 0, (best_doy - start_doy + 1)).astype(np.uint8)
    
    saveGeoTiff(os.path.join(out_dir, f"{os.path.basename(out_dir)}.DOY.tif"), 
                relative_doy.compute(), template_file=template_path)
    

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
        "--stat",
        help="min, max, or percentile",
        type=str,
        default="max",
    )
    parse.add_argument(
        "--percentile_value",
        help="percentile value (0-100) if stat is percentile",
        type=int,
        default=50,
    )
    parse.add_argument(
        "--output_dir", 
        help="Directory in which to save output", 
        required=True
    )
    parse.add_argument(
        "--search_source",
        help="Either STAC or earthaccess to search for HLS granules",
        type=str,
        default="earthaccess",
    )
    parse.add_argument(
        "--access_type",
        help="Either external (from http) or direct (from S3) to search for HLS granules",
        type=str,
        default="direct",
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
        stat=args.stat,
        percentile_value=int(args.percentile_value),
        save_dir=output_dir,
        search_source=args.search_source,
        access_type=args.access_type,
    )
