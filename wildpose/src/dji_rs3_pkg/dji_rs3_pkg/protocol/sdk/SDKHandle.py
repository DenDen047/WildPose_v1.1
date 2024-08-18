# encoding:utf-8

import sys
import os
import abc
import time
import queue
import binascii
import threading
import struct
from collections import namedtuple
from ctypes import *

from protocol.sdk import SDKCRC

import sys_infor

class SDKDataHandle(threading.Thread):
    def __init__(self, can_connection):
        super(SDKDataHandle, self).__init__()
        self.stopped = False
        self.dev = can_connection
        self.rsps = []
        self.rdcontent_lock = threading.Lock()

        self.cmd_callbacks = {
            '0E:00': self.cmd_callback_posControl,
            '0E:01': self.cmd_callback_speedControl,
            '0E:02': self.cmd_callback_getGimbalInfo,
            '0E:03': self.cmd_callback_setAngleLimit,
            '0E:04': self.cmd_callback_getAngleLimit,
            '0E:05': self.cmd_callback_setMotorStrength,
            '0E:06': self.cmd_callback_getMotorStrength,
            '0E:07': self.cmd_callback_setPush,
            '0E:08': self.cmd_callback_pushData,
            '0D:00': self.cmd_callback_cameraControl,
            '0D:01': self.cmd_callback_focusControl,
        }

    def __check_head_crc(self, head_data):
        cmd_prefix = ":".join(head_data[:10])
        pack_crc = ":".join(head_data[-2:])
        crc = SDKCRC.calc_crc(cmd_prefix, 16)
        if crc.upper() == pack_crc.upper():
            return True
        else:
            return False

    def __check_pack_crc(self, pack_data):
        cmd_prefix = ":".join(pack_data[:-4])
        pack_crc = ":".join(pack_data[-4:])
        crc = SDKCRC.calc_crc(cmd_prefix, 32)
        if crc.upper()  == pack_crc.upper():
            return True
        else:
            return False


    def run(self):
        v1_pack_list = []
        pack_len = 0
        step = 0
        canid_raw_str = ""
        canid_str = ""
        while not self.stopped:
            frame = self.dev.tunnel.recv_queue_handle.get()
            for data in frame:
                if step == 0:
                    if data == "AA":
                        v1_pack_list.append(data)
                        step = 1

                elif step == 1:
                    pack_len = int(data, 16)
                    v1_pack_list.append(data)
                    step = 2

                elif step == 2:
                    pack_len |= ((int(data, 16) & 0x3) << 8)
                    v1_pack_list.append(data)
                    step = 3

                elif step == 3:
                    v1_pack_list.append(data)
                    if len(v1_pack_list) == 12:
                        if self.__check_head_crc(v1_pack_list):
                            step = 4
                        else:
                            step = 0
                            v1_pack_list = []

                elif step == 4:
                    v1_pack_list.append(data)

                    if len(v1_pack_list) == pack_len:
                        step = 0
                        if self.__check_pack_crc(v1_pack_list):
                            self.__process_cmd(v1_pack_list)
                        v1_pack_list = []
                else:
                    step = 0
                    v1_pack_list = []

            time.sleep(0.1)
    def __process_cmd(self, data):
        string_data = ''
        for string in data:
            string_data += string
        hex_list = binascii.a2b_hex(string_data)
        # print('pack_list-->{}'.format(data))
        # print('string_data--->{}'.format(string_data))
        # print('hex_list--->{}'.format(hex_list))

        cmd_type = data[3]
        cmd_key = ''

        if cmd_type == '20':
            # 如果是应答帧，需要查下对应的发送命令

            cmd_data = self.dev.lastSendData

            # print('last data-->{}'.format(cmd_data))
            # print('hex_list--->{}'.format(hex_list))

            if cmd_data[8:10] == hex_list[8:10]:
                # print('Found a ACK packet {} for cmd {}'.format(hex_list, cmd_data))
                send_pack_list = ['{:02X}'.format(i) for i in cmd_data]
                # print('send pack --->{}'.format(send_pack_list))
                pack_data = send_pack_list
                cmd_key = '{}:{}'.format(send_pack_list[12], send_pack_list[13])
        else:
            pack_data = data
            cmd_key = '{}:{}'.format(data[12], data[13])

        if cmd_key != '' and cmd_key in self.cmd_callbacks.keys():
            self.cmd_callbacks[cmd_key](pack_data[12:-4])

    # 手持云台位置控制（CmdSet = 0x0E CmdID = 0x00）
    def cmd_callback_posControl(self, data):
        print('posControl-->{}'.format(data))

    # 手持云台速度控制（CmdSet = 0x0E CmdID = 0x01）
    def cmd_callback_speedControl(self, data):
        print('speedControl-->{}'.format(data))

    # 获取手持云台信息（CmdSet = 0x0E CmdID = 0x02）
    def cmd_callback_getGimbalInfo(self, data):
        print('getGimbalInfo-->{}'.format(data))

    # 设置手持云台限位角度（CmdSet = 0x0E CmdID = 0x03）
    def cmd_callback_setAngleLimit(self, data):
        print('setAngleLimit-->{}'.format(data))

    # 获取手持云台限位角度（CmdSet = 0x0E CmdID = 0x04）
    def cmd_callback_getAngleLimit(self, data):
        print('getAngleLimit-->{}'.format(data))

    # 设置手持云台电机力度（CmdSet = 0x0E CmdID = 0x05）
    def cmd_callback_setMotorStrength(self, data):
        print('setMotorStrength-->{}'.format(data))

    # 获取手持云台电机力度（CmdSet = 0x0E CmdID = 0x06）
    def cmd_callback_getMotorStrength(self, data):
        print('getMotorStrength-->{}'.format(data))

    # 手持云台参数推送设置（CmdSet = 0x0E CmdID = 0x07）
    def cmd_callback_setPush(self, data):
        print('setPush-->{}'.format(data))

    # 手持云台参数推送（CmdSet = 0x0E CmdID = 0x08）
    def cmd_callback_pushData(self, data):
        cmd_data = data[2:]
        string_data = ''
        for s in cmd_data:
            string_data += s
        hex_data = binascii.a2b_hex(string_data)
        ctrl_byte = eval('0X' + cmd_data[0])
        if (ctrl_byte >> 0) & 0x1 == 0:
            print("角度信息（姿态角、关节角）无效")
        else:
            print("角度信息（姿态角、关节角）有效")

        if (ctrl_byte >> 1) & 0x1 == 0:
            print("角度限位信息无效")
        else:
            print("角度限位信息有效")

        if (ctrl_byte >> 2) & 0x1 == 0:
            print("电机力度信息无效")
        else:
            print("电机力度信息有效")

        yaw_attitude_angle, roll_attitude_angle, pitch_attitude_angle, yaw_joint_angle, roll_joint_angle, pitch_joint_angle  = struct.unpack('6h', hex_data[1:13])
        print('Attitude Angle:')
        print('YAW: {:3.1f}, ROW: {:3.1f}, PITCH: {:3.1f}'.format(yaw_attitude_angle * 0.1, roll_attitude_angle * 0.1,
                                                                  pitch_attitude_angle * 0.1))
        print('Joint Angle:')
        print('YAW: {:3.1f}, ROW: {:3.1f}, PITCH: {:3.1f}'.format(yaw_joint_angle * 0.1, roll_joint_angle * 0.1,
                                                                  pitch_joint_angle * 0.1))

        sys_infor.gimbal_control_attitude_pitch = pitch_attitude_angle * 0.1
        sys_infor.gimbal_control_attitude_roll = roll_attitude_angle * 0.1
        sys_infor.gimbal_control_attitude_yaw = yaw_attitude_angle * 0.1
        sys_infor.gimbal_control_joint_pitch = pitch_joint_angle * 0.1
        sys_infor.gimbal_control_joint_roll = roll_joint_angle * 0.1
        sys_infor.gimbal_control_joint_yaw = yaw_joint_angle * 0.1


    # 相机简单控制指令（包含拍照 录像 停止录像）（CmdSet = 0x0D CmdID = 0x00）
    def cmd_callback_cameraControl(self, data):
        print('cameraControl-->{}'.format(data))

    # 相机焦距设置指令（CmdSet = 0x0D CmdID = 0x01）
    def cmd_callback_focusControl(self, data):
        print('focusControl-->{}'.format(data))