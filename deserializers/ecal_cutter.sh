#!/bin/bash

output_dir=/mnt/data/WildPose_v1.1/2022-12-03
if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi
ecal_meas_cutter \
    -i /media/ikuta/Expansion/2022-12-03/2022-09-08_12-01-54.990_wildpose_v1.1 \
    -o $output_dir \
    --config ecal_cutter_config.yml