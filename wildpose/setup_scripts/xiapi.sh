
#!/bin/bash
# source: https://www.ximea.com/support/wiki/apis/Linux_TX1_and_TX2_Support#Installing-XIMEA-API-package

sudo apt update
sudo apt install ca-certificates -y

cd ~/Downloads
mkdir tmp
cd ~/Downloads/tmp
wget https://www.ximea.com/downloads/recent/XIMEA_Linux_SP.tgz
tar -xf XIMEA_Linux_SP.tgz
cd package
./install
cd ~/Downloads
rm -rf tmp

sudo gpasswd -a "$(whoami)" plugdev
if [ -f /etc/rc.local ]
then
    sudo sed -i '/^exit/ d' /etc/rc.local
else
    echo '#!/bin/sh -e'                                            | sudo tee    /etc/rc.local > /dev/null
fi
echo 'echo 0 > /sys/module/usbcore/parameters/usbfs_memory_mb' | sudo tee -a /etc/rc.local > /dev/null
echo 'exit 0'                                                  | sudo tee -a /etc/rc.local > /dev/null
sudo chmod a+x /etc/rc.local
#enable controlling of memory frequency by user
echo 'KERNEL=="emc_freq_min", ACTION=="add", GROUP="plugdev", MODE="0660"' | sudo tee /etc/udev/rules.d/99-emc_freq.rules > /dev/null
#optional: allow user to use realtime priorities
sudo groupadd -fr realtime
echo '*         - rtprio   0' | sudo tee    /etc/security/limits.d/ximea.conf > /dev/null
echo '@realtime - rtprio  81' | sudo tee -a /etc/security/limits.d/ximea.conf > /dev/null
echo '*         - nice     0' | sudo tee -a /etc/security/limits.d/ximea.conf > /dev/null
echo '@realtime - nice   -16' | sudo tee -a /etc/security/limits.d/ximea.conf > /dev/null
sudo gpasswd -a "$(whoami)" realtime
sudo mkdir /etc/systemd/system/user@.service.d
echo '[Service]'                                                                                 | sudo tee    /etc/systemd/system/user@.service.d/cgroup.conf > /dev/null
echo 'PermissionsStartOnly=true'                                                                 | sudo tee -a /etc/systemd/system/user@.service.d/cgroup.conf > /dev/null
echo 'ExecStartPost=-/bin/sh -c "echo 950000 > /sys/fs/cgroup/cpu/user.slice/cpu.rt_runtime_us"' | sudo tee -a /etc/systemd/system/user@.service.d/cgroup.conf > /dev/null
#reboot
sudo reboot