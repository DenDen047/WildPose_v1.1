import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d
# import pyvista as pv

from utils.file_loader import load_config_file, load_pcd
from utils.format_conversion import get_timestamp_from_pcd_fpath


class KeyEvent:
    def __init__(self, pcd_fpaths,
                 timestamps, pcd_mode, init_geometry=None):
        self.pcd_fpaths = pcd_fpaths
        self.timestamps = timestamps
        self.pcd_idx = 0
        self.pcd_mode = pcd_mode
        self.current_pcd = init_geometry

        self.body_heights = [0] * len(self.pcd_fpaths)

    def update_pcd(self, vis):
        # reset the scene
        viewpoint_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        if self.current_pcd is not None:
            self.current_pcd.points = o3d.utility.Vector3dVector([])
            self.current_pcd.colors = o3d.utility.Vector3dVector([])

        # load new pcd file
        pcd = load_pcd(self.pcd_fpaths[self.pcd_idx], mode=self.pcd_mode)

        # update the scene
        self.current_pcd = pcd
        vis.add_geometry(self.current_pcd)
        vis.get_view_control().convert_from_pinhole_camera_parameters(viewpoint_param)

        self.increment_pcd_index()
        return True

    def increment_pcd_index(self,):
        self.pcd_idx += 10
        if len(self.pcd_fpaths) <= self.pcd_idx:
            self.pcd_idx %= len(self.pcd_fpaths)


def main():
    # args
    config_fpath = 'config.hjson'
    pcd_mode = 'open3d'

    # load the config file
    config = load_config_file(config_fpath)

    # get file paths of textured point cloud
    pcd_fpaths = sorted(glob.glob(
        os.path.join(config['scene_folder'], 'lidar', '*.pcd')))
    timestamps = sorted([
        get_timestamp_from_pcd_fpath(f)
        for f in pcd_fpaths
    ])
    assert len(pcd_fpaths) == len(timestamps)

    # prepare the open3d viewer
    init_geometry = load_pcd(pcd_fpaths[0], mode=pcd_mode)
    event_handler = KeyEvent(
        pcd_fpaths,
        timestamps,
        pcd_mode=pcd_mode,
        init_geometry=init_geometry
    )
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    # register key callback functions with GLFW_KEY
    vis.register_key_callback(32, event_handler.update_pcd)  # space
    vis.add_geometry(init_geometry)
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.7, 0.7, 0.7])
    vis.poll_events()
    vis.run()

    vis.destroy_window()


if __name__ == '__main__':
    main()
