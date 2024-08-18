# WildPose v1.1

## Hardware

### XIMEA -- MQ022CG-CM

![Slide86](https://user-images.githubusercontent.com/6120047/187175093-c170c1db-6820-45db-b62d-7cf7d2296982.jpeg)

## Prerequisite

- JetPack v5
- ROS2 Foxy

## Setup

### Jetson AGX Xavier Developer Kit

#### JetPack

We used a [Jetson AGX Xavier Developer Kit](https://developer.nvidia.com/embedded/jetson-agx-xavier-developer-kit).
To set up the Jetson AGX Xavier Developer Kit, we need a host computer installed Ubuntu 20.04.

1. Download and install [NVIDIA SDK Manager](https://developer.nvidia.com/nvidia-sdk-manager) onto the host computer.
2. Connect the host computer and Jetson AGX Xavier Developer Kit (see this [video](https://www.youtube.com/watch?v=-nX8eD7FusQ)).
3. Run NVIDIA SDK Manager and follow the instruction.

Unleash the limitation of CPU ([the reference](https://forums.developer.nvidia.com/t/cpus-usage-problem-solved/65993/3)).
```bash
$ sudo nvpmodel -m 0
$ sudo /usr/bin/jetson_clocks
```

#### General Settings

- Settings > Power > Blank Screen > `Never`

#### Network Configuration

On the Jetson, you can use `nmcli` command to change the network settings.

Set the static IP on the UCT network ([reference](https://f1tenth.readthedocs.io/en/stable/getting_started/software_setup/optional_software_nx.html)).
```bash
$ nmcli c show
NAME                UUID                                  TYPE      DEVICE
Wired connection 1  b72f3d20-4de2-3d44-9c45-9689d79f22e4  ethernet  eth0
docker0             bcb6f95d-5cf5-483d-ac09-c312a4da8c0b  bridge    docker0
$ sudo nmcli c modify "Wired connection 1" ipv4.address [NEW_ADDRESS]/27
```

#### SSH

***HOST COMPUTER***

For the ssh login from your computer, you should make a pair of ssh key on the host computer.
```bash
$ ssh-keygen -t rsa
```

Then, copy the public key into the Jetson.
```bash
$ ssh-copy-id -i ~/.ssh/wildpose_jetsonagx.pub [user]@[ip address]
```

Add the jetson IP address information in `~/.ssh/config`:
```
Host [ip address]
    HostName [ip address]
    User naoya
    IdentityFile ~/.ssh/wildpose_jetsonagx
    UseKeychain yes
    AddKeysToAgent yes
```

#### dotfiles

Set the dotfiles you wanna use (e.g., [Naoya's dotfiles](https://github.com/DenDen047/dotfiles)).

#### ROS2 Foxy

Let's install [ROS2 Foxy](https://docs.ros.org/en/foxy/index.html) following with [the official guide](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html).
See `setup_scripts/ros2_foxy.sh`.

```bash
$ cd setup_scripts
$ ./ros2_foxy.sh
```

#### The Official Ximea Camera Driver

This is [the original GitHub repository](https://github.com/wavelab/ximea_ros_cam) and [the Guide for Jetson](https://www.ximea.com/support/wiki/apis/Linux_TX1_and_TX2_Support#Installing-XIMEA-API-package).

```bash
$ cd setup_scripts/
$ chmod +x ./xiapi.sh
$ ./xiapi.sh
# reopen the shell
```

##### Setup the USB FS Memory Max Allocation to Infinite

This is done to make sure that the USB FS buffering size is sufficient for high bandwidth streams through USB 3.0

*Set this with every new shell*:
Put `echo 0 > /sys/module/usbcore/parameters/usbfs_memory_mb` into `/etc/rc.local`

Or

*Apply to current shell*:
`echo "0" | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb`

#### ROS2 Packages

To show the ximea camera image data, you are recommended to install [image_view](https://index.ros.org/p/image_view/).

```bash
$ sudo apt install -y ros-foxy-image-view
$ sudo apt install -y ros-foxy-camera-info-manager
```

#### XIMEA Camera Driver for ROS2

```bash
$ mkdir ~/ros2_ws
$ cd ~/ros2_ws
$ git clone git@github.com:African-Robotics-Unit/ximea_ros2_cam.git
$ cd ~/ros2_ws/ximea_ros2_cam/
$ git fetch
$ git checkout -b develop
$ sudo apt install -y ros-foxy-camera-info-manager
$ colcon build --packages-select cam_driver_pkg
$ colcon build --packages-select cam_bringup --symlink-install

# Test the camera driver
$ ros2 launch cam_bringup cam_test.launch.py
```

Then, add `source ~/ros2_ws/ximea_ros2_cam/install/setup.bash` into `~/.bashrc`.

To avoid the [error 45](https://github.com/Fu-physics/Ximea/blob/master/xiPython/v3/ximea/xidefs.py#L49), you have to run the following command.

```bash
$ sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb >/dev/null <<<0
```

#### Livox-SDK

This is [the original GitHub repository](https://github.com/Livox-SDK/Livox-SDK).

```bash
$ sudo apt install -y cmake gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
$ cd ~/Documents
$ git clone https://github.com/Livox-SDK/Livox-SDK.git
$ cd Livox-SDK
```

Edit `Livox-SDK/sdk_core/CMakeLists.txt` by adding `-fPIC`.
Ref: https://github.com/Livox-SDK/livox_ros2_driver/issues/9#issuecomment-1048578018

```bash
$ cd build && cmake .. -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
$ make
$ sudo make install
$ source /opt/ros/foxy/setup.bash
```

Set the network configuration:
- IP address: `192.168.1.2`
- Netmask: `255.255.255.0`
- Gatway: `192.168.1.1`

#### Livox ROS2 Driver

This is [the original GitHub repository](https://github.com/Livox-SDK/livox_ros2_driver).

```bash
$ cd ~/ros2_ws
$ git clone https://github.com/Livox-SDK/livox_ros2_driver.git livox_ros2_driver/src
$ cd livox_ros2_driver
$ colcon build
$ source ~/ros2_ws/livox_ros2_driver/install/setup.bash
```

Add `source ~/ros2_ws/livox_ros2_driver/install/setup.bash` into `~/.bashrc`.
Don't forget to change **the config file**.

#### jtop

To check the Jetson status, [`jtop`](https://github.com/rbonghi/jetson_stats) should be installed.
```bash
$ sudo -H pip install -U jetson-stats
$ sudo reboot
```

#### CAN bus setting

```bash
$ sudo apt install -y busybox
```

Add the following code into your `/etc/rc.local`:
```bash
sh /home/naoya/WildPose_v1.1/src/dji_rs3_pkg/enable_CAN.sh &
```

References:
- [Enabling CAN on Nvidia Jetson Xavier Developer Kit](https://medium.com/@ramin.nabati/enabling-can-on-nvidia-jetson-xavier-developer-kit-aaaa3c4d99c9)
- [hmxf/can_xavier -- GitHub](https://github.com/hmxf/can_xavier)

#### Remote Desktop

```bash
$ sudo apt install -y xrdp
$ cd
$ echo "xfce4-session" | tee .xsession
$ sudo reboot
```


### Host Computer

To develop ROS2 programs on your host/local computer, VS Code ROS Extension was used.
Please refer to see the following video:

<iframe width="560" height="315" src="https://www.youtube.com/embed/teA20AjBlG8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


#### VSCode

Recommend Extensions:
- [C/C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack)
- [CMake](https://marketplace.visualstudio.com/items?itemName=twxs.cmake)
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [ROS2](https://marketplace.visualstudio.com/items?itemName=nonanonno.vscode-ros2)


## Usage

Run WildPose with the following commmand, and the data will be recorded in a rosbag file in `rosbags/`.

```bash
$ ros2 launch wildpose_bringup wildpose_launch.py
```

## Build

```bash
$ cd ~/WildPose_v1.1
$ colcon build --packages-select wildpose_bringup --symlink-install
```

## Generate Video

Install requirements:

```bash
$ sudo apt install -y ffmpeg
$ sudo apt install -y python3-roslib python3-rospy python3-sensor-msgs python3-opencv
$ pip3 install tqdm ffmpeg-python
```

Convert a rosbag file into a video file:

```bash
$ cd src/
$ ./rosbag2video.py --input_db ~/WildPose_v1.1/rosbags/20221002_192302/20221002_192302_0.db3
```