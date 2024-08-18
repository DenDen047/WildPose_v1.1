import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory
import launch
from launch import LaunchDescription
from launch_ros.actions import Node


################### Livox TELE-15 user-defined parameters start ###################
xfer_format   = 0    # 0-Pointcloud2(PointXYZRTL), 1-customized pointcloud format
multi_topic   = 0    # 0-All LiDARs share the same topic, 1-One LiDAR one topic
data_src      = 0    # 0-lidar,1-hub
publish_freq  = 10.0 # freqency of publish,1.0,2.0,5.0,10.0,etc
return_mode = 2 # 2-dual return
output_type   = 0
frame_id      = 'livox_frame'
lvx_file_path = '/home/livox/livox_test.lvx'
cmdline_bd_code = 'livox0000000001'

cur_path = os.path.split(os.path.realpath(__file__))[0] + '/'
cur_config_path = cur_path + '../config'
user_config_path = os.path.join(cur_config_path, 'livox_lidar_config.json')
################### Livox TELE-15 user-defined parameters end #####################

livox_ros2_params = [
    {"xfer_format": xfer_format},
    {"multi_topic": multi_topic},
    {"data_src": data_src},
    {"publish_freq": publish_freq},
    {"output_data_type": output_type},
    {"frame_id": frame_id},
    {"lvx_file_path": lvx_file_path},
    {"return_mode": return_mode},
    {"user_config_path": user_config_path},
    {"cmdline_input_bd_code": cmdline_bd_code}
]


ximea_cam_parameters = {
    ################### XIMEA camera user-defined parameters start ###################
    ####################
    # General Configuration Parameters Go Here!
    ####################

    # camera properties (https://github.com/wavelab/ximea_ros_cam)
    'serial_no': "29970951",  # serial number on the backplate
    'cam_name': "ximea_MQ022CG-CM",   # Name of the camera used when saving camera images and snapshots under the directory pointed by image_directory
    # 'calib_file': "", # Calibration file used by the camera
    'frame_id': '0',
    'num_cams_in_bus': 1, # Number of USB cameras processed by a single USB controller
    'bw_safetyratio': 1.0,  # Bandwidth safety ratio, a multiplier to the bandwidth allocated for each camera
    'poll_time': 2.0, # Used to set the duration (in seconds) which the camera is attempted to be opened again.
    'poll_time_frame': 1/170, # the ROS timer loop period (in seconds) for the ximea camera node. It should generally be set to a rate that is a factor higher than the camera capture rate.
    'publish_xi_image_info': True,    # Flag for publishing the extra ximea camera information provided with each image acquisition.

    # directory to save images (make sure that directory exists and that it is an absolute path).
    'image_directory': "/home/dev/",    # must be absolute path, not relative path (i.e. '~')

    # save images to disk, under the directory `<image_directory>/stream`
    'save_disk': False,

    # Saves images everytime a trigger is pressed, under the director `<image_directory>/calib`
    'calib_mode': False,
    'cam_context_path': '~/WildPose_v1.1/wildpose/record/cam_context_{}.bin'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')),

    ####################
    # Diagnostics Configuration Parameters Go Here!
    ####################

    'enable_diagnostics': True,
    'data_age_max': 0.1,
    'pub_frequency': 10.0,
    'pub_frequency_tolerance': 1.0,

    ####################
    # Camera Configuration Parameters Go Here!
    ####################

    # # image_transport compressed image parameters
    # 'image_transport_compressed_format': "png", # jpg or png
    # 'image_transport_compressed_jpeg_quality': 100, # 1 to 100 (1: min quality)
    # 'image_transport_compressed_png_level': 5,  # 1 to 9 (9: max compression)

    # colour image format
    'format': "XI_RAW8",

    # camera coloring
    # white balance mode: 0 - none, 1 - use coefficients, 2: auto
    'white_balance_mode': 0,
    'white_balance_coef_red': 3.0,  # white balance red coefficient (0 to 8)
    'white_balance_coef_green': 0.0,    # white balance green coefficient (0 to 8)
    'white_balance_coef_blue': 4.0, # white balance blue coefficient (0 to 8)

    # triggering (0 - none, 1 - software trigger (NOT IMPLMENTED YET), 2 - hardware trigger)
    'cam_trigger_mode': 0,
    'hw_trigger_edge': 0,   # if hw trigger, 0/1: rising/falling edge trigger

    # for camera frame rate
    'frame_rate_control': True, # enable or disable frame rate control (works if no triggering is enabled)
    'frame_rate_set': 170,   # for trigger mode, fps limiter (0 for none)
    'img_capture_timeout': 1000,    # timeout in milliseconds for xiGetImage()

    # exposure settings
    'auto_exposure': False,          # auto exposure on or off
    'exposure_time': 1000,           # manual exposure time in microseconds
    'manual_gain': 5.0,              # manual exposure gain (dB)
    'auto_exposure_priority': 0.8,   # auto exposure to gain ratio (1.0: favour only exposure)
    'auto_time_limit': 30000,        # auto exposure time limit in microseconds
    'auto_gain_limit': 2.0,          # auto exposure gain limit

    # region of interest
    # MQ022CG-CM: 2048x1088
    # - 1080p(1920x1080)
    # 'roi_left': 64,      # top left corner in pixels
    # 'roi_top': 4,
    # 'roi_width': 1280,  # width height in pixels
    # 'roi_height': 1024,
    # - 720p (1280x720)
    'roi_left': 512,      # top left corner in pixels
    'roi_top': 184,
    'roi_width': 1280,  # width height in pixels
    'roi_height': 720,
    ################### XIMEA camera user-defined parameters end #####################
}


def generate_launch_description():
    now = datetime.now()

    ximea_cam_driver = Node(
        package='cam_driver_pkg',
        executable='ximea_ros2_cam_node',
        name='ximea_cam_publisher',
        output='screen',
        parameters=[{k: v} for k, v in ximea_cam_parameters.items()],
        arguments=['--ros-args', '--log-level','ERROR']
    )

    image_viewer = Node(
        package='image_view',
        executable='image_view',
        name='image_view',
        output='screen',
        remappings=[
            ("/image", "/image_raw"),
        ],
        on_exit=launch.actions.Shutdown()
    )

    livox_driver = Node(
        package='livox_ros2_driver',
        executable='livox_ros2_driver_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=livox_ros2_params,
        arguments=['--ros-args', '--log-level','ERROR']
    )

    rosbag = launch.actions.ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '/xi_image_info', '/image_raw', '/livox/lidar', '/livox/imu',
            '--qos-profile-overrides-path', '~/WildPose_v1.1/wildpose/src/wildpose_bringup/config/reliability_override.yaml',
            # '--polling-interval', '0',
            '-o', os.path.join('./rosbags/', now.strftime('%Y%m%d_%H%M%S')),
        ],
        output='screen',
        on_exit=launch.actions.Shutdown()
    )

    return LaunchDescription([
        ximea_cam_driver,
        image_viewer,
        livox_driver,
        rosbag,
    ])