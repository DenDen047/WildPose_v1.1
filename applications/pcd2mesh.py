import os
import numpy as np
import pyvista as pv
import open3d as o3d


def main():
    pcd_fpath = './data/martial_eagle_stand/textured_pcds/coloured_accumulation.pcd'
    xlim = (-100, 100)
    ylim = (-100, 100)
    zlim = (18, 19)
    # pcd_fpath = './data/giraffe_stand/textured_pcds/coloured_accumulation.pcd'
    # xlim = (-100, 100)
    # ylim = (-100, 100)
    # zlim = (85, 95)

    # load data
    pcd = o3d.io.read_point_cloud(pcd_fpath)
    points = np.asarray(pcd.points)  # [N, 3]

    # filter 3d points
    mask = (points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1]) & \
        (points[:, 1] >= ylim[0]) & (points[:, 1] <= ylim[1]) & \
        (points[:, 2] >= zlim[0]) & (points[:, 2] <= zlim[1])

    filtered_points = points[mask]

    cloud = pv.PolyData(filtered_points)
    # cloud.plot()

    volume = cloud.delaunay_3d(alpha=0.015)
    mesh = volume.extract_geometry()
    mesh.save(os.path.join(
        os.path.dirname(pcd_fpath),
        'mesh.stl'
    ))
    mesh.plot()

if __name__ == '__main__':
    main()
