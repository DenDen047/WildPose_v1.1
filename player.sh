#!/bin/sh
tmux new-session -d
tmux split-window -v 'ros2 run image_view image_view --ros-args --remap /image:=/image_raw'
tmux split-window -h 'ecal_play_gui'
tmux select-pane -t 1
tmux split-window -h 'rviz2 --display-config /home/naoya/ros2_ws/livox_ros2_driver/src/livox_ros2_driver/config/livox_lidar.rviz'
tmux -2 attach-session -d