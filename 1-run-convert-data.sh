#!/bin/bash

python src/convert_data.py \
    --source pdf \
    --dest yaml \
    --in_dir parlament/pdfs \
    --out_dir parlament/yamls \
    --process_xml


