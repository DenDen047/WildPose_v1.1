#!/bin/bash
# https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/HR/ControllerAreaNetworkCan.html

sudo busybox devmem 0x0c303010 w 0x400
sudo busybox devmem 0x0c303018 w 0x458

sudo modprobe can
sudo modprobe can_raw
sudo modprobe mttcan

sudo ip link set can0 type can bitrate 1000000 dbitrate 1000000 berr-reporting on fd on
sudo ip link set up can0

# sudo ip link set down can0


exit 0
