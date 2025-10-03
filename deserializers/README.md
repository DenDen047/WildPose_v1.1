![logo](docs/resources/ARU_logo_rectangle.png)

# eCAL Deserializers


## Description
This repo contains deserializers for [eCAL](https://github.com/eclipse-ecal/rmw_ecal) measurements of four ROS2 messages used in our [M2S2](m2s2_ws) drivers.

The deserializers extract the raw data stored in the .hdf5 files and converts it into desirable data according to their message definitons.

Messages:
1. standard [ROS2](https://docs.ros.org/en/foxy/index.html) sensor_msgs/msg/Image
2. custom M2S2 flir_boson_interfaces/msg/ThermalRaw
3. custom M2S2 bme280_interfaces/msg/EnviroData
4. custom M2S2 radar_interfaces/msg/Frame

## Install

[eCAL](https://github.com/eclipse-ecal/ecal)
```bash
sudo add-apt-repository ppa:ecal/ecal-5.13
sudo apt update
sudo apt install -y ecal
```

[ROS2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
```bash
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

sudo apt update && sudo apt upgrade -y
sudo apt install -y ros-humble-desktop ros-dev-tools
mkdir -p ~/ros2_ws/src

# add the below command to .bashrc
source /opt/ros/humble/setup.bash

rosdep init
rosdep update
```

[rmw_ecal](https://github.com/eclipse-ecal/rmw_ecal)
```bash
cd ~/ros2_ws/src
sudo apt install -y \
    ros-humble-osrf-testing-tools-cpp \
    ros-humble-test-msgs \
    ros-humble-image-view
git clone https://github.com/eclipse-ecal/rosidl_typesupport_protobuf.git
git clone https://github.com/eclipse-ecal/rmw_ecal
sudo apt install -y libprotobuf-dev protobuf-compiler
colcon build --symlink-install --cmake-args -DBUILD_TESTING=OFF

# add the below commands to ~/.bashrc
export RMW_IMPLEMENTATION=rmw_ecal_dynamic_cpp
source ~/ros2_ws/src/install/setup.bash
```

## Trim eCAL measurement file

Usually, eCAL data is super large.
So, we recommend to [trim the eCAL file](https://eclipse-ecal.github.io/ecal/stable/applications/meas_cutter/meas_cutter.html) before deserialisation.

First, you need to check the timestamp on eCAL player and ROS2 `image_view`:
```bash
ros2 run image_view image_view --ros-args --remap /image:=/image_raw
```

Then edit the `deserializers/ecal_cutter_config.yml` and run eCAL cutter:
```bash
cd deserializers
vim ecal_cutter_config.yml
./ecal_cutter.sh
```

Or you can cut scenes at once:
```bash
cd deserializers
python batch_ecal_cutter.py
```

Note: the input and output arguments of `ecal_meas_cutter` should be folder paths.

## Usage

First of all, you have build the whole folder.
```bash
$ cmake .
```

### Ximea Camera Deserializer

This [deserializer](src/deserialize_ximea.cpp) converts a recorded ecal measurement of type [sensor_msgs/msg/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html) into RGB .jpeg images.

<b>NOTE:</b> Ximea API's demosaic color defect correction is not optimized for ARM processors, and so performance may degrade significantly. To avoid this, we captured raw 8-bits or 16-bits images (by setting the <i>format</i> parameter to <b>RAW8</b> or <b>RAW16</b> in the config file of the [M2S2 ROS2 driver](add link)) and we post-process later using the API's [offline processing](https://www.ximea.com/support/wiki/apis/XiAPI_Offline_Processing), by saving the camera context each time data is collected.

[xiAPI](https://www.ximea.com/support/wiki/apis/XIMEA_Linux_Software_Package)
```bash
# download package
wget  https://updates.ximea.com/public/ximea_linux_sp_beta.tgz
tar xzf ximea_linux_sp_beta.tgz
cd package

# install
sudo apt-get update
sudo apt-get install build-essential linux-headers-"$(uname -r)" 
sudo apt-get install libtiff5
./install
```

build with:
```bash
$ cmake --build . --target ximea
```

run:
```bash
./ecal_sample_ximea 'meas_folder_path' 'channel_name' 'cam_context_path' 'out_path_rgb' 'out_path_raw'
```

where:
- `meas_folder_path` is the path to the input eCAL measurement folder
- `channel_name` is the channel you wish to deserialize. (This is often `rt/image_raw`)
- `cam_context_path` is the path to where the camera context file is saved for a specific measurement, to perform post-processing of the raw ximea images
- `out_path_rgb` is the path to where you wish to store the post-processed RGB images
- `out_path_raw` is the path to where you wish to store the post-processed RAW images


### Livox Deserializer

This [deserializer](src/deserialize_livox.cpp) converts a recorded ecal measurement of type [sensor_msgs/msg/PointCloud2](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud2.html) into a pcd file using the [Point Cloud Library](https://github.com/PointCloudLibrary/pcl).

dependencies:
```bash
$ sudo apt install -y libpcl-dev
$ sudo apt install -y python3-pcl pcl-tools
$ git clone git@github.com:PointCloudLibrary/pcl.git
$ cd pcl
$ mkdir build
$ cd build
$ cmake ..
$ make -j8
$ sudo make install
```

build with:
```bash
$ cmake --build . --target livox_imu lidar
```

run:
```bash
$ ./ecal_sample_lidar 'meas_folder_path' 'channel_name' 'out_path'
```

where:
- `meas_folder_path` is the path to the input eCAL measurement folder
- `channel_name` is the channel you wish to deserialize. (This is often `rt/livox/lidar`)
- `out_path` is the folder path to where you wish to store the pcd files

The IMU data structure:
```msg
std_msgs/Header header
    int32 sec
    int32 nanosec
    string frame_id

geometry_msgs/Quaternion orientation
    float64 x   # Default: 0
    float64 y   # Default: 0
    float64 z   # Default: 0
    float64 w   # Default: 1
float64[9] orientation_covariance   # Row major about x, y, z axes

geometry_msgs/Vector3 angular_velocity
    float64 x
    float64 y
    float64 z
float64[9] angular_velocity_covariance  # Row major about x, y, z axes

geometry_msgs/Vector3 linear_acceleration
    float64 x
    float64 y
    float64 z
float64[9] linear_acceleration_covariance   # Row major x, y z
```

### BME280 Deserializer

This [deserializer](src/deserialize_bme280.cpp) converts a recorded ecal measurement of type [bme280_interfaces/msg/EnviroData](m2s2_ws/src/bme280_ROS2_driver/bme280_interfaces/msg/EnviroData.msg) into a JSON file with the following fields for each measurement: frame_id, timestamp_secs, timestamp_nanosecs, temperature, pressure, humidity.

build with:
```bash
$ cmake --build . --target bme280
```

run:
```bash
./ecal_sample_bme280 'meas_path' 'channel_name' 'out_file_name'
```

where:
- `meas_path` is the path to the input eCAL measurement file
- `channel_name` is the channel you wish to deserialize. (This is often `rt/ros2_topic_name`)
- `out_file_name` is the name of the JSON file you wish to save the data to

### FLIR Boson Deserializer

This [deserializer](src/deserialize_flir_boson.cpp) converts a recorded ecal measurement of type [flir_boson_interfaces/msg/ThermalRaw](m2s2_ws/src/flir_boson_ROS2_driver/flir_boson_interfaces/msg/ThermalRaw.msg) into raw 16 bit .png images, as well as grayscale false-coloured .jpeg images.

build with:
```bash
$ cmake --build . --target boson
```

run:
```bash
./ecal_sample_boson 'meas_path' 'channel_name' 'out_path_raw' 'out_path_rgb'
```

where:
- `meas_path` is the path to the input eCAL measurement file
- `channel_name` is the channel you wish to deserialize. (This is often `rt/ros2_topic_name`)
- `out_path_raw` is the path to where you wish to store the 16 bit raw images
- `out_path_rgb` is the path to where you wish to store the false-coloured images

### Radar Deserializer

This [deserializer](src/deserialize_radar.cpp) converts a recorded ecal measurement of type [radar_interfaces/msg/Frame](m2s2_ws/src/radar_ROS2_driver/radar_interfaces/msg/Frame.msg) into a JSON file with the following fields for each radar frame: frame_id, timestamp_secs, timestamp_nanosecs, frame_size, frame_data.

build with:
```bash
$ cmake --build . --target radar
```

run:
```bash
./ecal_sample_radar 'meas_path' 'channel_name' 'out_file_name'
```

where:
- `meas_path` is the path to the input eCAL measurement file
- `channel_name` is the channel you wish to deserialize. (This is often `rt/ros2_topic_name`)
- `out_file_name` is the name of the JSON file you wish to save the data to