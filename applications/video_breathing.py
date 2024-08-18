import os
import sys
import cv2
import numpy as np
import pickle
import glob
import pandas as pd
from tqdm import tqdm
from scipy.signal import lombscargle, butter, filtfilt, detrend
from scipy.interpolate import interp1d

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import seaborn as sns
import scienceplots


from utils.file_loader import load_config_file


# plt.style.use(['science', 'nature', 'no-latex'])
# figure(figsize=(10, 6))
mpl.style.use('ggplot')
plt.rcParams.update({
    "pdf.fonttype": 42,
})
figure(figsize=(3.5 * 2, 2.5))   # max width is 3.5 for single column

CONFIG = {
    "scene_dir": "data/lion_sleep",
    "pcd_dir": "data/lion_sleep/lidar",
    "rgb_dir": "data/lion_sleep/rgb",
    "sync_rgb_dir": "data/lion_sleep/sync_rgb",
    "mask_dir": "data/lion_sleep/masks_lion",
    "textured_pcd_dir": "data/lion_sleep/textured_pcds",
    "bbox_info_fpath": "data/lion_sleep/train.json",
}


def normalize_data(data, new_min=-1, new_max=1):
    min_val = np.min(data)
    max_val = np.max(data)
    return ((data - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min


def main():
    config = CONFIG

    # load the data
    with open('saved_data.pkl', 'rb') as f:
        input_data = pickle.load(f)
    timestamps = np.array(input_data['timestamp'])
    data = input_data['data']
    img_fpaths = sorted(glob.glob(os.path.join(config['sync_rgb_dir'], '*.jpeg')))

    df = pd.read_excel(os.path.join(config['scene_dir'], 'body_state.xlsx'))
    labels = df['state']
    labels = labels.where(pd.notnull(labels), None).tolist()

    # set the start from t=0
    timestamps = timestamps - timestamps[0]

    # make the y values
    ys = [np.average(v) for v in data]

    # Interpolate ys onto a uniform grid
    interp_func = interp1d(timestamps, ys, kind='linear')
    uniform_timestamps = np.linspace(
        min(timestamps),
        max(timestamps),
        len(timestamps))
    uniform_ys = interp_func(uniform_timestamps)

    # Design a band-pass filter for the frequency range
    fs = 1 / np.mean(np.diff(uniform_timestamps))  # Sampling frequency
    lowcut = 0.75
    highcut = 1.4
    order = 6
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y_filtered = filtfilt(b, a, uniform_ys)

    # show the filtered plot
    # normalized_y_filtered = normalize_data(np.array(y_filtered))
    normalized_y_filtered = y_filtered * 1000

    # define plot settings
    fig = plt.figure(figsize=(18,6), tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 3])
    n_frames = len(uniform_timestamps)
    df = pd.DataFrame({
        'x': uniform_timestamps,
        'y': normalized_y_filtered,
    })
    xmin = 0
    xmax = np.max(uniform_timestamps)
    ymin = np.min(normalized_y_filtered)
    ymax = np.max(normalized_y_filtered)

    # make the animation
    def animate(i):
        # show the line plot
        plt.subplot(gs[0])
        data = df.iloc[:int(i+1)]
        plt.plot(data['x'], data['y'], color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Filtered Translation (mm)')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

        # show the corresponding video frame
        plt.subplot(gs[1])
        img_fpath = img_fpaths[i]
        img = cv2.cvtColor(cv2.imread(img_fpath), cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.grid(False)
        plt.axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)  # Adjust spacing as needed

    ani = animation.FuncAnimation(
        fig, animate,
        frames=tqdm(range(n_frames), file=sys.stdout),
        repeat=True)

    # save the animation
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('results/output.mp4', writer=writer)


if __name__ == '__main__':
    main()
