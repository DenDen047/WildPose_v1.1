import cv2
import numpy as np
import open3d as o3d
import quaternion
import glob
import os
import json
from tqdm import tqdm

from utils.file_loader import load_camera_parameters
from projection_functions import extract_rgb_from_image_pure
from utils.camera import make_intrinsic_mat, make_extrinsic_mat


CONFIG = {
    "scene_dir": "data/martial_eagle_stand",
    "pcd_dir": "data/martial_eagle_stand/lidar",
    "sync_rgb_dir": "data/martial_eagle_stand/sync_rgb",
    'texture_img_fpath': 'data/martial_eagle_stand/texture.jpeg',
    "textured_pcd_dir": "data/martial_eagle_stand/textured_pcds",
}
IMG_WIDTH, IMG_HEIGHT = 1280, 720

COLORS = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1,
        "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1,
        "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    {"color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"},
    {"color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"},
    {"color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"},
    {"color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"},
    {"color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"},
    {"color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"},
    {"color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door-stuff"},
    {"color": [218, 88, 184], "isthing": 0, "id": 118, "name": "floor-wood"},
    {"color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"},
    {"color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"},
    {"color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"},
    {"color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"},
    {"color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"},
    {"color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"},
    {"color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"},
    {"color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"},
    {"color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"},
    {"color": [106, 154, 176], "isthing": 0,
        "id": 145, "name": "playingfield"},
    {"color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"},
    {"color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"},
    {"color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"},
    {"color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"},
    {"color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"},
    {"color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"},
    {"color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"},
    {"color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"},
    {"color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"},
    {"color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"},
    {"color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"},
    {"color": [137, 54, 74], "isthing": 0, "id": 171, "name": "wall-brick"},
    {"color": [135, 158, 223], "isthing": 0, "id": 175, "name": "wall-stone"},
    {"color": [7, 246, 231], "isthing": 0, "id": 176, "name": "wall-tile"},
    {"color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wall-wood"},
    {"color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water-other"},
    {"color": [183, 121, 142], "isthing": 0,
        "id": 180, "name": "window-blind"},
    {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"},
    {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"},
    {"color": [190, 153, 153], "isthing": 0,
        "id": 185, "name": "fence-merged"},
    {"color": [146, 139, 141], "isthing": 0,
        "id": 186, "name": "ceiling-merged"},
    {"color": [70, 130, 180], "isthing": 0,
        "id": 187, "name": "sky-other-merged"},
    {"color": [134, 199, 156], "isthing": 0,
        "id": 188, "name": "cabinet-merged"},
    {"color": [209, 226, 140], "isthing": 0,
        "id": 189, "name": "table-merged"},
    {"color": [96, 36, 108], "isthing": 0,
        "id": 190, "name": "floor-other-merged"},
    {"color": [96, 96, 96], "isthing": 0,
        "id": 191, "name": "pavement-merged"},
    {"color": [64, 170, 64], "isthing": 0,
        "id": 192, "name": "mountain-merged"},
    {"color": [152, 251, 152], "isthing": 0,
        "id": 193, "name": "grass-merged"},
    {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"},
    {"color": [206, 186, 171], "isthing": 0,
        "id": 195, "name": "paper-merged"},
    {"color": [152, 161, 64], "isthing": 0,
        "id": 196, "name": "food-other-merged"},
    {"color": [116, 112, 0], "isthing": 0,
        "id": 197, "name": "building-other-merged"},
    {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"},
    {"color": [102, 102, 156], "isthing": 0,
        "id": 199, "name": "wall-other-merged"},
    {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"},
]
colors_indice = [0, 5, 10, 15, 20, 25, 30, 35, 13, 45, 50, 55, 60, 65, 70]

KEYPOINTS = {   # giraffe_stand
    '000_004.jpeg': {
        'nose': [839, 159],
        'r_eye': [774, 113],
        'neck': [713, 499],
        'hip': [552, 559],
    },
    '001_005.jpeg': {
        'nose': [838, 159],
        'r_eye': [776, 114],
        'neck': [710, 497],
        'hip': [555, 557],
    },
    '002_006.jpeg': {
        'nose': [836, 160],
        'r_eye': [777, 115],
        'neck': [709, 497],
        'hip': [552, 561],
    },
    '003_007.jpeg': {
        'nose': [837, 159],
        'r_eye': [778, 113],
        'neck': [711, 499],
        'hip': [551, 555],
    },
    '004_008.jpeg': {
        'nose': [839, 160],
        'r_eye': [777, 114],
        'neck': [713, 500],
        'hip': [551, 562],
    },
    '005_009.jpeg': {
        'nose': [841, 162],
        'r_eye': [776, 115],
        'neck': [713, 497],
        'hip': [552, 559],
    },
    '006_010.jpeg': {
        'nose': [841, 161],
        'r_eye': [778, 115],
        'neck': [713, 504],
        'hip': [552, 558],
    },
    '007_011.jpeg': {
        'nose': [839, 160],
        'r_eye': [780, 114],
        'neck': [711, 503],
        'hip': [552, 558],
    },
    '008_012.jpeg': {
        'nose': [841, 163],
        'r_eye': [780, 116],
        'neck': [713, 497],
        'hip': [557, 559],
    },
    '009_013.jpeg': {
        'nose': [839, 160],
        'r_eye': [777, 114],
        'neck': [711, 505],
        'hip': [548, 561],
    },
    '010_014.jpeg': {
        'nose': [839, 163],
        'r_eye': [779, 117],
        'neck': [712, 503],
        'hip': [542, 558],
    },
    '011_015.jpeg': {
        'nose': [840, 163],
        'r_eye': [779, 118],
        'neck': [712, 500],
        'hip': [548, 563],
    },
    '012_016.jpeg': {
        'nose': [840, 164],
        'r_eye': [778, 116],
        'neck': [713, 505],
        'hip': [547, 558],
    },
    '013_017.jpeg': {
        'nose': [838, 164],
        'r_eye': [777, 118],
        'neck': [710, 497],
        'hip': [547, 559],
    },
    '014_018.jpeg': {
        'nose': [840, 166],
        'r_eye': [777, 119],
        'neck': [710, 501],
        'hip': [548, 565],
    },
    '015_019.jpeg': {
        'nose': [840, 162],
        'r_eye': [777, 118],
        'neck': [709, 500],
        'hip': [550, 563],
    },
    '016_020.jpeg': {
        'nose': [838, 162],
        'r_eye': [776, 117],
        'neck': [711, 503],
        'hip': [546, 561],
    },
}
KEYPOINTS = {   # martial_eagle_stand
    '000_004.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '001_005.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '002_006.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '003_007.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '004_008.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '005_009.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '006_010.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '007_011.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '008_012.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '009_013.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '010_014.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '011_015.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '012_016.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '013_017.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '014_018.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '015_019.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
    '016_020.jpeg': {
        'nose': [700, 132],
        'r_eye': [641, 120],
        'neck': [604, 239],
        'hip': [82, 412],
    },
}


def lidar2cam_projection(pcd, extrinsic):
    tmp = np.insert(pcd, 3, 1, axis=1).T
    tmp = np.delete(tmp, np.where(tmp[0, :] < 0), axis=1)
    pcd_in_cam = extrinsic.dot(tmp)

    return pcd_in_cam


def cam2image_projection(pcd_in_cam, intrinsic):
    pcd_in_image = intrinsic.dot(pcd_in_cam)
    pcd_in_image[:2] /= pcd_in_image[2, :]

    return pcd_in_image


def sync_lidar_and_rgb(lidar_dir, rgb_dir):
    rgb_fpaths = sorted(glob.glob(os.path.join(rgb_dir, '*.jpeg')))
    lidar_fpaths = sorted(glob.glob(os.path.join(lidar_dir, '*.pcd')))

    rgb_list = []
    lidar_list = []

    for rgb_fpath in rgb_fpaths:
        rgb_filename = os.path.basename(rgb_fpath).split('.')[0]
        second_rgb, decimal_rgb = rgb_filename.split('_')[1:3]

        second_rgb = int(second_rgb)
        decimal_rgb = float('0.' + decimal_rgb)
        rgb_timestamp = second_rgb + decimal_rgb

        diff_list = []
        lidar_fp_list = []
        for lidar_fpath in lidar_fpaths:
            lidar_filename = os.path.basename(lidar_fpath).split('.')[0]
            _, _, second_lidar, decimal_lidar = lidar_filename.split('_')
            second_lidar = int(second_lidar)
            decimal_lidar = float('0.' + decimal_lidar)

            lidar_timestamp = second_lidar + decimal_lidar
            diff = abs(rgb_timestamp - lidar_timestamp)

            diff_list.append(diff)
            lidar_fp_list.append(lidar_fpath)

        diff_list = np.array(diff_list)
        matching_lidar_file = lidar_fp_list[np.argmin(diff_list)]

        rgb_list.append(rgb_fpath)
        assert os.path.exists(matching_lidar_file)
        lidar_list.append(matching_lidar_file)

    return lidar_list, rgb_list


def get_2D_gt(gt_json_path):

    gt_json = json.load(open(gt_json_path, 'r'))
    img_dict = {}

    annotations = gt_json['annotations']
    images = gt_json['images']

    for idx in range(len(annotations)):

        anno = annotations[idx]
        bbox = anno['bbox']
        img_id = anno['image_id']
        obj_id = anno['category_id']
        img_filename = os.path.basename(images[img_id]['file_name'])

        if img_filename not in img_dict:
            img_dict[img_filename] = []
        img_dict[img_filename].append([bbox, obj_id])
    return img_dict


def load_rgb_img(fpath: str):
    bgr_img = cv2.imread(fpath)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img


def main(accumulation=False):
    # arguments
    data_dir = CONFIG['scene_dir']
    lidar_dir = CONFIG['pcd_dir']
    rgb_dir = CONFIG['sync_rgb_dir']
    texture_img_fpath = CONFIG['texture_img_fpath']
    calib_fpath = os.path.join(data_dir, 'manual_calibration.json')
    output_dir = CONFIG['textured_pcd_dir']

    fx, fy, cx, cy, rot_mat, translation = load_camera_parameters(calib_fpath)

    if accumulation:
        # load the texture image
        lidar_list = sorted(glob.glob(os.path.join(lidar_dir, '*.pcd')))
        rgb_img = load_rgb_img(texture_img_fpath)

        # mark the keypoints on the image
        for img_name in KEYPOINTS:
            for body_part in KEYPOINTS[img_name]:
                cv2.circle(
                    rgb_img, tuple(KEYPOINTS[img_name][body_part]),
                    radius=5,
                    color=(255, 0, 0),
                    thickness=-1)

        # accumulate all the point cloud
        accumulated_pcd_in_lidar = None
        for pcd_fpath in lidar_list:
            pcd_in_lidar = o3d.io.read_point_cloud(pcd_fpath)
            pcd_points = np.asarray(pcd_in_lidar.points)  # [N, 3]

            if accumulated_pcd_in_lidar is None:
                accumulated_pcd_in_lidar = pcd_points
            else:
                accumulated_pcd_in_lidar = np.vstack((accumulated_pcd_in_lidar, pcd_points))


        # load the camera parameters
        intrinsic = make_intrinsic_mat(fx, fy, cx, cy)
        extrinsic = make_extrinsic_mat(rot_mat, translation)

        # project the point cloud to camera and its image sensor
        pcd_in_cam = lidar2cam_projection(accumulated_pcd_in_lidar, extrinsic)
        pcd_in_img = cam2image_projection(pcd_in_cam, intrinsic)
        pcd_in_cam = pcd_in_cam.T[:, :-1]
        pcd_in_img = pcd_in_img.T[:, :-1]

        pcd_colors, valid_mask_save = extract_rgb_from_image_pure(
            pcd_in_img, rgb_img, width=IMG_WIDTH, height=IMG_HEIGHT)
        pcd_with_rgb = np.concatenate([pcd_in_cam, pcd_colors], 1)
        pcd_with_rgb = pcd_with_rgb[valid_mask_save]  # [N, 6]
        textured_pcd = o3d.geometry.PointCloud()
        textured_pcd.points = o3d.utility.Vector3dVector(
            pcd_with_rgb[:, :3])
        textured_pcd.colors = o3d.utility.Vector3dVector(
            pcd_with_rgb[:, 3:])

        # visualize
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.add_geometry(textured_pcd)
        opt = vis.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.7, 0.7, 0.7])
        vis.poll_events()
        vis.run()
        vis.destroy_window()

        o3d.io.write_point_cloud(
            os.path.join(output_dir, 'coloured_accumulation_with_keypoints.pcd'),
            textured_pcd)
    else:
        lidar_list, rgb_list = sync_lidar_and_rgb(lidar_dir, rgb_dir)
        for idx in tqdm(range(len(rgb_list))):
            # load the frame image
            rgb_fpath = rgb_list[idx]
            rgb_img = load_rgb_img(rgb_fpath)
            # load point cloud of the frame
            lidar_path = lidar_list[idx]
            pcd_in_lidar = o3d.io.read_point_cloud(lidar_path)
            pcd_in_lidar = np.asarray(pcd_in_lidar.points)

            # load the camera parameters
            intrinsic = make_intrinsic_mat(fx, fy, cx, cy)
            extrinsic = make_extrinsic_mat(rot_mat, translation)

            # project the point cloud to camera and its image sensor
            pcd_in_cam = lidar2cam_projection(pcd_in_lidar, extrinsic)
            pcd_in_img = cam2image_projection(pcd_in_cam, intrinsic)

            pcd_in_cam = pcd_in_cam.T[:, :-1]
            pcd_in_img = pcd_in_img.T[:, :-1]

            pcd_colors, valid_mask_save = extract_rgb_from_image_pure(
                pcd_in_img, rgb_img, width=IMG_WIDTH, height=IMG_HEIGHT)
            pcd_with_rgb_save = np.concatenate([pcd_in_cam, pcd_colors], 1)
            pcd_with_rgb_save = pcd_with_rgb_save[valid_mask_save]  # [N, 6]
            textured_pcd = o3d.geometry.PointCloud()
            textured_pcd.points = o3d.utility.Vector3dVector(
                pcd_with_rgb_save[:, :3])
            textured_pcd.colors = o3d.utility.Vector3dVector(
                pcd_with_rgb_save[:, 3:])

            file_prefix = rgb_fpath.split('/')[-1].split('.')[0]
            o3d.io.write_point_cloud(
                os.path.join(output_dir, file_prefix + '.pcd'),
                textured_pcd)


if __name__ == '__main__':
    main(accumulation=True)