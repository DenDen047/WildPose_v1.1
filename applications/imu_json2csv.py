import os
import glob
import json
import pandas as pd
import numpy as np

from utils.format_conversion import get_timestamp_from_pcd_fpath


def main():
    data_dir = './data/springbok_herd2/'

    # load data
    lidar_fpaths = sorted(glob.glob(os.path.join(data_dir, 'lidar', '*.pcd')))
    imu_json_fpath = os.path.join(data_dir, 'imu.json')
    with open(imu_json_fpath) as f:
        imu_data = json.load(f)

    # get the timestamp range
    start_time = get_timestamp_from_pcd_fpath(lidar_fpaths[0])
    end_time = get_timestamp_from_pcd_fpath(lidar_fpaths[-1])

    # extract the IMU data within the time
    def imu_frame2timestamp(imu_frame):
        return float(
            str(imu_frame['timestamp_sec']) + '.' + str(imu_frame['timestamp_nanosec']))
    imu_frames = []
    for imu_frame in imu_data:
        timestamp = imu_frame2timestamp(imu_frame)
        if start_time <= timestamp <= end_time:
            imu_frames.append(imu_frame)

    # make the csv file
    column_names = \
        ['timestamp'] + \
        [f'gyro_{x} (rad/s)' for x in ['x', 'y', 'z']] + \
        [f'acc_{x} (g)' for x in ['x', 'y', 'z']]
    data = []
    for imu_frame in imu_frames:
        row = \
            [imu_frame2timestamp(imu_frame)] + \
            imu_frame['angular_velocity'] + \
            imu_frame['linear_acceleration']
        data.append(row)
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(os.path.join(data_dir, 'imu.csv'))


if __name__ == '__main__':
    main()
