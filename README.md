## Usage

### MAAP DPS

```python
from maap.maap import MAAP

maap = MAAP(maap_host="api.maap-project.org")

jobs = []
for tile in ["14VLQ", "18WXS", "16WFB", "26WMC", "19VDL"]:
    job = maap.submitJob(
        algo_id="HLSComposite",
        version="v0.2",
        identifier="test-run",
        queue="maap-dps-worker-64gb",
        tile=tile,
        start_datetime="2013-01-01",
        end_datetime="2025-10-31",
        out_dir="s3://maap-ops-workspace/shared/zhouqiang06/HLS_composite",
    )
    jobs.append(job)

```
