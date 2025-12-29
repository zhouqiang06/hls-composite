## Usage

### MAAP DPS

To run the algorithm via DPS, you can follow this example. Provide the S3 URI for an input spatial file using the `input_file` argument. This file will be read using `geopandas` then the records will be filtered down to the ones that intersect the HLS raster asset footprint for the selected MGRS tile.

```python
from maap.maap import MAAP

maap = MAAP(maap_host="api.maap-project.org")

jobs = []
for tile in ["14VLQ", "18WXS", "16WFB", "26WMC", "19VDL"]:
    job = maap.submitJob(
        algo_id="HLSPointTimeSeriesExtraction",
        version="v0.2",
        identifier="test-run",
        queue="maap-dps-worker-16gb",
        tile=tile,
        start_datetime="2013-01-01",
        end_datetime="2025-10-31",
        out_dir="s3://maap-ops-workspace/shared/zhouqiang06/HLS_composite",
    )
    jobs.append(job)

```
