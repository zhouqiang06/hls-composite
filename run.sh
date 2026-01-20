#!/usr/bin/env -S bash --login
set -euo pipefail
# This script is the one that is called by the DPS.
# Use this script to prepare input paths for any files
# that are downloaded by the DPS and outputs that are
# required to be persisted

# Get current location of build script
basedir=$(dirname "$(readlink -f "$0")")

# Create output directory to store outputs.
# The name is output as required by the DPS.
# Note how we dont provide an absolute path
# but instead a relative one as the DPS creates
# a temp working directory for our code.

mkdir -p output


# DPS downloads all files provided as inputs to
# this directory called input.
# In our example the image will be downloaded here.
# INPUT_DIR=input
OUTPUT_DIR=output

# input_filename=$(ls -d input/*)

# Parse positional arguments (3 required, 2 optional)
if [[ $# -lt 4 ]] || [[ $# -gt 5 ]]; then
    echo "Error: Expected 4-6 arguments, got $#"
    echo "Usage: $0 <mgrs_tile> <start_date> <end_date> <output_dir>"
    echo "  bands: space-separated list of band names (e.g., 'red green blue')"
    exit 1
fi

tile="$1"
start_date="$2"
end_date="$3"
stat="$4"
percentile_value="$5"
search_source="$6"
access_type="$7"

# Call the script using the absolute paths
# Use the updated environment when calling 'uv run'
# This lets us run the same way in a Terminal as in DPS
# Any output written to the stdout and stderr streams will be automatically captured and placed in the output dir

# unset PROJ env vars
unset PROJ_LIB
unset PROJ_DATA

# Build the command with required arguments
cmd=(
    uv run --no-dev "${basedir}/main.py"
    --tile "${tile}"
    --start_date "${start_date}"
    --end_date "${end_date}"
    --stat "${stat}"
    --percentile_value "${percentile_value}"
    --output_dir "${OUTPUT_DIR}"
    --search_source "${search_source}"
    --access_type "${access_type}"
)

# Execute the command with UV_PROJECT environment variable
UV_PROJECT="${basedir}" "${cmd[@]}"
