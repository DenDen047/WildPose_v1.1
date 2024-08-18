import cv2
import numpy as np
import open3d as o3d
import glob
import os
from tqdm import tqdm


CONFIG = {
    "scene_dir": "data/giraffe_stand",
    "rgb_dir": "data/giraffe_stand/sync_rgb",
    "pcd_dir": "data/giraffe_stand/lidar",
    "merged_rgb_dir": "data/giraffe_stand/merged_rgb",
    "merged_pcd_dir": "data/giraffe_stand/merged_pcd",
}
MERGE_SIZE = 5


def main():
    # arguments
    data_dir = CONFIG['scene_dir']
    rgb_dir = CONFIG['rgb_dir']
    lidar_dir = CONFIG['pcd_dir']
    output_rgb_dir = CONFIG['merged_rgb_dir']
    output_pcd_dir = CONFIG['merged_pcd_dir']
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_pcd_dir, exist_ok=True)

    # load the texture image
    rgb_list = sorted(glob.glob(os.path.join(rgb_dir, '*.jpeg')))
    lidar_list = sorted(glob.glob(os.path.join(lidar_dir, '*.pcd')))

    # accumulate all the point cloud
    for i in tqdm(range(len(lidar_list) - MERGE_SIZE + 1)):
        accumulated_points = None
        accumulated_imgs = None
        j = i + MERGE_SIZE

        # merge the image
        for rgb_fpath in rgb_list[i:j]:
            # NOTE: you need to write the parser of pcd files if you get the intensity.
            rgb_img = np.expand_dims(cv2.imread(rgb_fpath), axis=0)

            if accumulated_imgs is None:
                accumulated_imgs = rgb_img
            else:
                accumulated_imgs = np.concatenate((accumulated_imgs, rgb_img), axis=0)
        cv2.imwrite(
            os.path.join(
                output_rgb_dir,
                str(i).zfill(3) + '_' + str(j-1).zfill(3) + '.jpeg'
            ),
            np.mean(accumulated_imgs, axis=0)
        )

        for pcd_fpath in lidar_list[i:j]:
            # NOTE: you need to write the parser of pcd files if you get the intensity.
            pcd_in_lidar = o3d.io.read_point_cloud(pcd_fpath)
            pcd_points = np.asarray(pcd_in_lidar.points)  # [N, 3]

            if accumulated_points is None:
                accumulated_points = pcd_points
            else:
                accumulated_points = np.vstack((accumulated_points, pcd_points))

        # make the new pcd file
        output_pcd = o3d.geometry.PointCloud()
        output_pcd.points = o3d.utility.Vector3dVector(accumulated_points)

        o3d.io.write_point_cloud(
            os.path.join(
                output_pcd_dir,
                str(i).zfill(3) + '_' + str(j-1).zfill(3) + '.pcd'
            ),
            output_pcd)


if __name__ == '__main__':
    main()