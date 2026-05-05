#!/bin/bash

sbatch scripts/run_metadata_creation_multimodalhugs.sh 
    -i final_output/ \
    -s video_segments \
    -o metadata \
    -c 0.3 \
    -l es \
    --cased \
    --filter_sessions 
