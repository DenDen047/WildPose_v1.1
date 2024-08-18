import os
import glob
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.format_conversion import get_timestamp_from_pcd_fpath


def get_position(timestamp, pos_df):
    # Calculate the absolute difference between the input timestamp and
    # timestamps in the DataFrame
    abs_diff = np.abs(pos_df['time'] - timestamp)

    # Find the index of the row with the smallest absolute difference
    closest_index = abs_diff.idxmin()

    # Retrieve the 3D position data for that row
    closest_row = pos_df.iloc[closest_index]

    # Extract x, y, and z coordinates and return them as a list
    position = [closest_row['x'], closest_row['y'], closest_row['z']]

    return position


def main():
    data_dir = './data/springbok_herd2/'

    # load lidar timestamp
    csv_fpath = os.path.join(data_dir, 'trajectory_raw', 'lidar_frames.csv')
    df = pd.read_csv(csv_fpath, names=['file_name'], header=0, index_col=0)
    timestamps = np.array(df['file_name'].apply(
        get_timestamp_from_pcd_fpath).tolist())

    # load trajectory data
    all_csv_fpaths = sorted(
        glob.glob(
            os.path.join(
                data_dir,
                'trajectory_raw',
                '*.csv')))
    csv_fpaths = [
        f for f in all_csv_fpaths
        if re.fullmatch(r'\d+\.csv', os.path.basename(f))
    ]
    dfs = {}
    for csv_fpath in csv_fpaths:
        key = os.path.splitext(os.path.basename(csv_fpath))[0]
        df = pd.read_csv(
            csv_fpath,
            names=['time', 'x', 'y', 'z'], header=0
        )
        df = df.where(df != -1e-6, other=np.nan)
        df['time'] = timestamps
        dfs[key] = df

    # load other data
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
        imu_timestamp = imu_frame2timestamp(imu_frame)
        if start_time <= imu_timestamp <= end_time:
            imu_frames.append(imu_frame)

    # make the csv file
    for key in dfs.keys():
        column_names = \
            ['imu_timestamp (s)'] + \
            [f'pos_{x} (m)' for x in ['x', 'y', 'z']] + \
            [f'gyro_{x} (rad/s)' for x in ['x', 'y', 'z']] + \
            [f'acc_{x} (g)' for x in ['x', 'y', 'z']]
        data = []
        for imu_frame in imu_frames:
            imu_timestamp = imu_frame2timestamp(imu_frame)
            position = get_position(imu_timestamp, dfs[key])
            row = \
                [imu_timestamp] + \
                position + \
                imu_frame['angular_velocity'] + \
                imu_frame['linear_acceleration']
            data.append(row)
        df = pd.DataFrame(data, columns=column_names)
        df = df.sort_values(by=['imu_timestamp (s)'], ignore_index=True)
        plt.plot(df['imu_timestamp (s)'])
        df.to_csv(f'results/kalman_filter_data_{key}.csv')


if __name__ == '__main__':
    main()
