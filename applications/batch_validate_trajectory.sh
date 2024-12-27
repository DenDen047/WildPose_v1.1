#!/bin/bash
ROOT_DIR=/Users/ikuta/Documents/Projects/PhD/WildPose_v1.1/data/calibration_v1.2
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_14-31-30.321_measurement --ref_distance 20 --ref_radius 1 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_14-32-52.247_measurement --ref_distance 20 --ref_radius 2 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_14-36-31.564_measurement --ref_distance 40 --ref_radius 1 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_14-37-55.275_measurement --ref_distance 40 --ref_radius 2 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_14-42-08.824_measurement --ref_distance 60 --ref_radius 1 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_14-43-38.205_measurement --ref_distance 60 --ref_radius 2 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_14-45-17.671_measurement --ref_distance 60 --ref_radius 3 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_14-48-08.768_measurement --ref_distance 80 --ref_radius 1 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_14-50-22.893_measurement --ref_distance 80 --ref_radius 2 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_14-52-18.990_measurement --ref_distance 80 --ref_radius 3 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_14-58-39.732_measurement --ref_distance 100 --ref_radius 1 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_15-00-13.424_measurement --ref_distance 100 --ref_radius 2 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_15-02-59.674_measurement --ref_distance 100 --ref_radius 3 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_15-06-12.368_measurement --ref_distance 120 --ref_radius 1 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_15-07-44.596_measurement --ref_distance 120 --ref_radius 2 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_15-09-34.972_measurement --ref_distance 120 --ref_radius 3 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_15-15-30.672_measurement --ref_distance 140 --ref_radius 1 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_15-16-40.500_measurement --ref_distance 140 --ref_radius 2 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_15-18-23.431_measurement --ref_distance 140 --ref_radius 3 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_15-23-42.342_measurement --ref_distance 160 --ref_radius 1 --n_revolutions 3 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_15-25-53.645_measurement --ref_distance 160 --ref_radius 2 --n_revolutions 3 && \
python validate_trajectory.py --mode motion_2d --data_dir ${ROOT_DIR}/2024-05-26_15-28-32.795_measurement --ref_distance 160 --ref_radius 3