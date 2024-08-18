#!/bin/bash

# This script is based on [the official guide](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html).

# Set locale
sudo apt update && sudo apt install locales -y
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Setup Sources
sudo apt update && sudo apt install curl gnupg2 lsb-release -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update && sudo apt upgrade -y
sudo apt install ros-foxy-desktop -y

# Install Python3
sudo apt install python3-pip -y
pip3 install argcomplete

# Install colcon
sudo apt install python3-colcon-common-extensions -y
