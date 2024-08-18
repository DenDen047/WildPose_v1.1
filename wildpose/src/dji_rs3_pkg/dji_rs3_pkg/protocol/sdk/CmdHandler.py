#encoding:utf-8

import binascii
import struct

#手持云台位置控制（CmdSet = 0x0E CmdID = 0x00）
def cmd_callback_posControl(data):
	print('posControl-->{}'.format(data))

#手持云台速度控制（CmdSet = 0x0E CmdID = 0x01）
def cmd_callback_speedControl(data):
	print('speedControl-->{}'.format(data))

#获取手持云台信息（CmdSet = 0x0E CmdID = 0x02）
def cmd_callback_getGimbalInfo(data):
	print('getGimbalInfo-->{}'.format(data))

#设置手持云台限位角度（CmdSet = 0x0E CmdID = 0x03）
def cmd_callback_setAngleLimit(data):
	print('setAngleLimit-->{}'.format(data))

#获取手持云台限位角度（CmdSet = 0x0E CmdID = 0x04）
def cmd_callback_getAngleLimit(data):
	print('getAngleLimit-->{}'.format(data))

#设置手持云台电机力度（CmdSet = 0x0E CmdID = 0x05）
def cmd_callback_setMotorStrength(data):
	print('setMotorStrength-->{}'.format(data))

#获取手持云台电机力度（CmdSet = 0x0E CmdID = 0x06）
def cmd_callback_getMotorStrength(data):
	print('getMotorStrength-->{}'.format(data))

#手持云台参数推送设置（CmdSet = 0x0E CmdID = 0x07）
def cmd_callback_setPush(data):
	print('setPush-->{}'.format(data))

#手持云台参数推送（CmdSet = 0x0E CmdID = 0x08）
def cmd_callback_pushData(data):
	print('pushData-->{}'.format(data))
	cmd_data = data[2:]
	string_data = ''
	for s in cmd_data:
		string_data += s
	hex_data = binascii.a2b_hex(string_data)
	print('cmd_data-->{}'.format(cmd_data))
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

	yaw_attitude_angle,row_attitude_angle,pitch_attitude_angle,yaw_joint_angle,row_joint_angle,pitch_joint_angle = struct.unpack('6h',hex_data[1:13])
	print('Attitude Angle:')
	print('YAW: {:3.1f}, ROW: {:3.1f}, PITCH: {:3.1f}'.format(yaw_attitude_angle * 0.1, row_attitude_angle * 0.1, pitch_attitude_angle * 0.1))
	print('Joint Angle:')
	print('YAW: {:3.1f}, ROW: {:3.1f}, PITCH: {:3.1f}'.format(yaw_joint_angle * 0.1, row_joint_angle * 0.1, pitch_joint_angle * 0.1))

#相机简单控制指令（包含拍照 录像 停止录像）（CmdSet = 0x0D CmdID = 0x00）
def cmd_callback_cameraControl(data):
	print('cameraControl-->{}'.format(data))

#相机焦距设置指令（CmdSet = 0x0D CmdID = 0x01）
def cmd_callback_focusControl(data):
	print('focusControl-->{}'.format(data))


cmd_callbacks = {
	'0E:00':cmd_callback_posControl,
	'0E:01':cmd_callback_speedControl,
	'0E:02':cmd_callback_getGimbalInfo,
	'0E:03':cmd_callback_setAngleLimit,
	'0E:04':cmd_callback_getAngleLimit,
	'0E:05':cmd_callback_setMotorStrength,
	'0E:06':cmd_callback_getMotorStrength,
	'0E:07':cmd_callback_setPush,
	'0E:08':cmd_callback_pushData,
	'0D:00':cmd_callback_cameraControl,
	'0D:01':cmd_callback_focusControl,
}