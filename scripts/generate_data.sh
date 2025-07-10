#!/bin/bash

OUTPUT_DIR="../data"
mkdir -p "$OUTPUT_DIR"

for edp_file in *.edp; do
    base_name=$(basename "$edp_file" .edp)
    FreeFem++ "$edp_file" "$OUTPUT_DIR" 2>&1
done