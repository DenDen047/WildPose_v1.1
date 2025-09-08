#!/bin/bash
# source: https://www.ximea.com/support/wiki/apis/Linux_TX1_and_TX2_Support#Installing-XIMEA-API-package

sudo apt update
sudo apt install -y ca-certificates udev
/lib/systemd/systemd-udevd --daemon

wget https://www.ximea.com/downloads/recent/XIMEA_Linux_SP.tgz
tar -xf XIMEA_Linux_SP.tgz
cd package
./install
