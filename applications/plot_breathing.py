import os
import numpy as np
import pickle
import json
import pandas as pd
from scipy.signal import lombscargle, butter, filtfilt, detrend
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scienceplots

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.file_loader import load_config_file


plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams.update({
    "pdf.fonttype": 42,
})

CONFIG = {
    "scene_dir": "data/lion_sleep",
    "pcd_dir": "data/lion_sleep/lidar",
    "sync_rgb_dir": "data/lion_sleep/sync_rgb",
    "mask_dir": "data/lion_sleep/masks_lion",
    "textured_pcd_dir": "data/lion_sleep/textured_pcds",
    "bbox_info_fpath": "data/lion_sleep/train.json",
    "imu_fpath": "data/lion_sleep/imu.json",
}


def normalize_data(data, new_min=-1, new_max=1):
    min_val = np.min(data)
    max_val = np.max(data)
    return ((data - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min


def main(mode):
    config = CONFIG

    # load the data
    with open(os.path.join(config['scene_dir'], 'saved_data_lion_sleep.pkl'), 'rb') as f:
        input_data = pickle.load(f)
    timestamps = np.array(input_data['timestamp'])
    data = input_data['data']
    with open(os.path.join(config['scene_dir'], 'saved_data.pkl'), 'rb') as f:
        input_data_ground = pickle.load(f)
    timestamps_ground = np.array(input_data_ground['timestamp'])
    data_ground = input_data_ground['data']

    df = pd.read_excel(os.path.join(config['scene_dir'], 'body_state.xlsx'))
    labels = df['state']
    labels = labels.where(pd.notnull(labels), None).tolist()

    # set the start from t=0
    timestamps = timestamps - timestamps[0]

    # make the y values
    ys = [np.mean(v) for v in data]
    ys_ground = [np.median(v) for v in data_ground]

    # Interpolate ys onto a uniform grid
    interp_func = interp1d(timestamps, ys, kind='linear')
    uniform_timestamps = np.linspace(
        min(timestamps),
        max(timestamps),
        len(timestamps))
    uniform_ys = interp_func(uniform_timestamps)

    interp_func_ground = interp1d(timestamps_ground, ys_ground, kind='linear')
    uniform_timestamps_ground = np.linspace(
        min(timestamps_ground),
        max(timestamps_ground),
        len(timestamps_ground))
    uniform_ys_ground = interp_func_ground(uniform_timestamps_ground)

    # Design a band-pass filter for the frequency range
    fs = 1 / np.mean(np.diff(uniform_timestamps))  # Sampling frequency
    lowcut = 1
    highcut = 2
    # lowcut = 0.75
    # highcut = 1.4
    order = 6
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    # band-pass filter
    b, a = butter(order, [low, high], btype='band')
    y_filtered = filtfilt(b, a, uniform_ys)
    y_filtered_ground = filtfilt(b, a, uniform_ys_ground)

    if mode == 'origin':
        # show the original plot
        plt.plot(timestamps, ys)
        # plt.title('Original Plot')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
    elif mode == 'filtered':
        figure(figsize=(3.5 * 2, 2.5))   # max width is 3.5 for single column

        # show the filtered plot
        plt.plot(uniform_timestamps, y_filtered * 1000)
        ta = None
        tb = None
        color = None
        for i, t in enumerate(timestamps):
            lbl = labels[i]
            if (lbl == 1 or lbl == -1) and color is None:
                ta = timestamps[i - 1]
                if labels[i] == 1:
                    color = 'red'
                elif labels[i] == -1:
                    color = 'blue'
            elif color == 'red' and lbl != 1:
                tb = timestamps[i]
            elif color == 'blue' and lbl != -1:
                tb = timestamps[i]
            else:
                continue

            if ta is not None and tb is not None and color is not None:
                plt.axvspan(ta, tb, color=color, alpha=0.3, linewidth=0)
                ta = None
                tb = None
                color = None
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Filtered Translation (mm)', fontsize=10)
    elif mode == 'subtract':
        plt.plot(uniform_timestamps, uniform_ys - uniform_ys_ground)
        ta = None
        tb = None
        color = None
        for i, t in enumerate(timestamps):
            lbl = labels[i]
            if (lbl == 1 or lbl == -1) and color is None:
                ta = timestamps[i - 1]
                if labels[i] == 1:
                    color = 'red'
                elif labels[i] == -1:
                    color = 'blue'
            elif color == 'red' and lbl != 1:
                tb = timestamps[i]
            elif color == 'blue' and lbl != -1:
                tb = timestamps[i]
            else:
                continue

            if ta is not None and tb is not None and color is not None:
                plt.axvspan(ta, tb, color=color, alpha=0.3, linewidth=0)
                ta = None
                tb = None
                color = None
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Filtered Translation (mm)', fontsize=10)
    elif mode == 'spectral':
        # calculate the Lomb-Scargle periodogram
        f = np.linspace(0.01, 5, 1000)  # Frequency range
        pgram = lombscargle(np.array(timestamps), np.array(ys), f * 2 * np.pi, normalize=True)
        pgram_ground = lombscargle(np.array(timestamps_ground), np.array(ys_ground), f * 2 * np.pi, normalize=True)

        # show the spectral plot
        plt.plot(f, pgram, label='Lion Sleep')
        plt.plot(f, pgram_ground, label='Background')
        plt.legend()
        plt.axvspan(lowcut, highcut, color='yellow', alpha=0.5, linewidth=0)
        plt.xlabel('Frequency [Hz]', fontsize=10)
        plt.ylabel('Normalized amplitude', fontsize=10)
    elif mode == 'fft':
        figure(figsize=(4.5, 1.7))   # max width is 3.5 for single column

        # calculate the FFT
        f = np.linspace(0.01, 5, 1000)  # Frequency range
        N = len(uniform_ys)
        T = 1 / fs
        yf = fft(uniform_ys)
        xf = fftfreq(N, T)[:N//2]

        # show the spectral plot
        # plt.semilogy(xf, 2.0/N * np.abs(yf[0:N//2]))
        plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))

        # Highlight the range of bandpass filter
        plt.axvspan(lowcut, highcut, color='yellow', alpha=0.3, label='Bandpass Filter Range')

        plt.xlabel('Frequency (Hz)', fontsize=10)
        plt.ylabel('Amplitude', fontsize=10)
        plt.legend(fontsize=8)
    elif mode == 'imu':
        figure(figsize=(4.5, 2.0))  # Adjust as needed

        imu_data = json.load(open(config['imu_fpath'], 'r'))
        accs = []
        ang_vels = []
        timestamps = []
        for packet in imu_data:
            acc = packet['linear_acceleration']
            ang_vel = packet['angular_velocity']
            timestamp = packet['timestamp_sec'] + \
                packet['timestamp_nanosec'] / 1e9
            accs.append(acc)
            ang_vels.append(ang_vel)
            timestamps.append(timestamp)

        accs = np.array(accs)
        ang_vels = np.array(ang_vels)
        timestamps = np.array(timestamps)

        f = np.linspace(0.01, nyquist, 10000)  # Frequency range

        # Plot PSD for linear acceleration and angular velocity
        pgram = lombscargle(timestamps, accs[:, 2], f * 2 * np.pi)
        # pgram = lombscargle(timestamps, ang_vels[:, 2], f * 2 * np.pi)
        plt.plot(f, pgram, linewidth=0.8)

        # Highlight the range of bandpass filter
        plt.axvspan(lowcut, highcut, color='yellow',
                    alpha=0.3, label='Bandpass Filter Range')

        plt.xlabel('Frequency (Hz)', fontsize=10)
        plt.ylabel('PSD', fontsize=10)
        # plt.legend(fontsize=6, loc='upper right')
        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.tight_layout()
    else:
        raise ValueError(f"Invalid mode: {mode}")

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    for fmt in ['svg', 'pdf']:
        plt.savefig(f"results/output.{fmt}", format=fmt, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main('filtered')
