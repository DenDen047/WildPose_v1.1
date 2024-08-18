# encoding:utf-8

import sys
import os
import abc
import time
import queue
import binascii
import threading
from collections import namedtuple
from ctypes import *
from protocol.sdk.SDKHandle import *

import sys_infor

STATUS_OK = 1
STATUS_ERR = 0

NORMAL_SEND = 0
SINGLE_SEND = 1

DATA_FRAME = 0
REMOTE_FRAME = 1

STD_FRAME = 0
EXT_FRAME = 1

FRAME_LEN = 8

SET_BAUDRATE_CMD = 0
SET_FILTER_CMD = 1
START_FILTER_CMD = 2
CLEAR_FILTER_CMD = 3
SET_OVER_TIME = 4
SET_AUTO_SEND = 5
CLEAR_AUTO_SEND = 6

USBCAN_II_TYPE = 4
USBCAN_2E_U_TYPE = 21


SendIdx = 4
RecvIdx = 5
SetIdx = 9
CmdIdIdx = 10

class CANConnection():
    def __init__(self, send_id, recv_id,can_name,tunnel_type=USBCAN_II_TYPE,tunnel_id=0, can_index=0):

        # 将 .dll 库的依赖库添加到系统搜索路径
        os.environ["path"] += os.getcwd() + "\protocol\\tool"

        self.device = CanDev(can_name,tunnel_type,tunnel_id, can_index)
        self.tunnel = self.device.tunnel
        self.tunnel.set_recv_id_list([recv_id])
        self.set_send_id(send_id)

        self.lastSendData = b''

        # 变量
        self.stopped = False

        print('init can')
        self._init_can()
        print('start can')
        self._start_can()
        print('listen thread')
        self._listen_thread()

    def stop(self):
        self.stopped = True

    def _init_can(self):
        self.tunnel.init_can()

    def set_send_id(self, send_id):
        self.send_id = send_id

    def _start_can(self):
        self.tunnel.clear_buffer()
        self.tunnel.start_can()


    def _listen_thread(self):

        self.recv_thread = RecvCanData(self.tunnel)
        self.pack_thread = SDKDataHandle(self)
        self.recv_thread.setDaemon(True)
        self.pack_thread.setDaemon(True)

        self.recv_thread.start()
        self.pack_thread.start()

    def pop_recv_cmd(self, key=''):
        recv_cmd = self.tunnel.recv_queue[key].copy()
        for i in range(len(recv_cmd)):
            self.tunnel.recv_queue[key].pop(0)
        return recv_cmd

    def get_recv_cmd(self, key=''):
        # self.logger.info('sender: %s' % sender)
        if key in self.tunnel.recv_queue:
            return self.tunnel.recv_queue[key]
        return []

    def send_cmd(self, cmd):
        """
        cmd的格式跟之前v1的类似， "55:0d:04:33:0a:04:01:00:40:00:01:a2:21" 之类的
        """
        data = [int(i, 16) for i in cmd.split(":")]
        self.lastSendData = data

        status  =False
        if self.tunnel.send_data(self.send_id, data) == STATUS_ERR:
            raise sys_infor.WRITE_FAILED

    def close(self):
        self.device.close()

class VCI_INIT_CONFIG(Structure):
    _fields_ = [
        ("AccCode", c_long),
        ("AccMask", c_long),
        ("Reserved", c_long),
        ("Filter", c_ubyte),
        ("Timing0", c_ubyte),
        ("Timing1", c_ubyte),
        ("Mode", c_ubyte)
    ]

class VCI_CAN_OBJ(Structure):
    _fields_ = [
        ("ID", c_uint),
        ("TimeStamp", c_uint),
        ("TimeFlag", c_ubyte),
        ("SendType", c_ubyte),
        ("RemoteFlag", c_ubyte),
        ("ExternFlag", c_ubyte),
        ("DataLen", c_ubyte),
        ("Data", c_ubyte*8),
        ("Reserved", c_ubyte*3),
    ]

class VCI_CAN_STATUS(Structure):
    _fields_ = [
        ("ErrInterrupt", c_ubyte),
        ("regMode", c_ubyte),
        ("regStatus", c_ubyte),
        ("regALCapture", c_ubyte),
        ("regECCapture", c_ubyte),
        ("regEWLimit", c_ubyte),
        ("regRECounter", c_ubyte),
        ("regTECounter", c_ubyte),
        ("Reserved", c_long)
    ]

class VCI_ERR_INFO(Structure):
    _fields_ = [
        ("ErrCode", c_uint),
        ("Passive_ErrData", c_ubyte*3),
        ("ArLost_ErrData", c_ubyte)
    ]

class VCI_FILTER_RECORDER(Structure):
    _fields_ = [
        ("ExtFrame", c_long),
        ("Start", c_long),
        ("End", c_long)
    ]

P_VCI_INIT_CONFIG = POINTER(VCI_INIT_CONFIG)
P_VCI_ERR_INFO    = POINTER(VCI_ERR_INFO)
P_VCI_FILTER_RECORDER = POINTER(VCI_FILTER_RECORDER)


class CanTunnel(object):
    __metaclass__  = abc.ABCMeta
    def __init__(self, can_dev, can_index):
        self.dev = can_dev
        self.dev_type  = can_dev.dev_type
        self.dev_index = can_dev.dev_index
        self.can_name  = can_dev.name
        self.can_index = c_int(can_index)
        self.read_content = b''
        self.recv_filter = {}
        self.recv_queue={}
        self.recv_queue_canid = {}
        self.recv_queue_handle = queue.Queue()
        self.recv_num = {}
        self.__is_running = False
        self.recv_id_list = []

    @abc.abstractmethod
    def init_can(self, can_id):
        pass

    def set_recv_id_list(self, recv_id_list):
        self.recv_id_list = recv_id_list

    def clear_buffer(self):

        if self.can_name == "ZLG_USBCAN":
            ret = DLL.VCI_ClearBuffer(self.dev_type, self.dev_index, self.can_index)
        elif self.can_name == "GC_USBCAN":
            ret = DLL.ClearBuffer(self.dev_type, self.dev_index, self.can_index)

        if ret != STATUS_OK:
            raise sys_infor.CAN_CLEAN_FAILED

    def start_can(self):
        if self.dev.is_open:
            if self.can_name == "ZLG_USBCAN":
                ret = DLL.VCI_StartCAN(self.dev_type, self.dev_index, self.can_index)
            elif self.can_name == "GC_USBCAN":
                ret = DLL.StartCAN(self.dev_type, self.dev_index, self.can_index)

            if ret == STATUS_OK:
                self.__is_running = True

            else:
                raise sys_infor.CAN_START_FAILED
            return ret

    def get_receive_num(self):
        if self.__is_running:

            if self.can_name == "ZLG_USBCAN":
                return DLL.VCI_GetReceiveNum(self.dev_type, self.dev_index, self.can_index)
            elif self.can_name == "GC_USBCAN":
                return DLL.GetReceiveNum(self.dev_type, self.dev_index, self.can_index)
        else:
            return 0

    def reset_can(self):
        if self.__is_running:

            if self.can_name == "ZLG_USBCAN":
                return DLL.VCI_ResetCAN(self.dev_type, self.dev_index, self.can_index)
            elif self.can_name == "GC_USBCAN":
                return DLL.ResetCAN(self.dev_type, self.dev_index, self.can_index)
        return 0

    def recv_data(self, length, wait_time = 400):
        if self.__is_running:
            recv_buf = (VCI_CAN_OBJ*length)()
            p_recv_buf = pointer(recv_buf)

            try:
                if self.can_name == "ZLG_USBCAN":
                    recv_len = DLL.VCI_Receive(self.dev_type, self.dev_index, self.can_index, p_recv_buf, c_int(length),c_int(wait_time))
                elif self.can_name == "GC_USBCAN":
                    recv_len = DLL.Receive(self.dev_type, self.dev_index, self.can_index, p_recv_buf, c_int(length),c_int(wait_time))

            except WindowsError as e:
                return None

            if recv_len:
                data_list = []
                can_data  = []
                for i in range(recv_len):
                    if recv_buf[i].ID in self.recv_id_list:
                        for n in range(recv_buf[i].DataLen):
                            can_data.append("%02X" % recv_buf[i].Data[n])
                data_list.append(can_data)

                return data_list
            else:
                return None
        else:
            #logger.info("dev is not running")
            return None



    def send_data(self, can_id, data):
        if self.__is_running:
            data_len = len(data)
            full_frame_num, left_len = divmod(data_len, FRAME_LEN)
            if left_len == 0:
                frame_num = full_frame_num
            else:
                frame_num = full_frame_num + 1

            send_buf = (VCI_CAN_OBJ*(frame_num))()

            data_offset = 0
            for i in range(full_frame_num):
                send_buf[i].ID = can_id
                send_buf[i].SendType = NORMAL_SEND
                send_buf[i].RemoteFlag = DATA_FRAME
                send_buf[i].ExternFlag = STD_FRAME
                send_buf[i].DataLen = FRAME_LEN

                for j in range(FRAME_LEN):
                    send_buf[i].Data[j] = data[data_offset+j]

                data_offset +=FRAME_LEN

            if left_len > 0:
                send_buf[frame_num - 1].ID = can_id
                send_buf[frame_num - 1].SendType = NORMAL_SEND
                send_buf[frame_num - 1].RemoteFlag = DATA_FRAME
                send_buf[frame_num - 1].ExternFlag = STD_FRAME
                send_buf[frame_num - 1].DataLen = left_len

                for j in range(left_len):
                    send_buf[frame_num - 1].Data[j] = data[data_offset+j]

            p_send_buf = pointer(send_buf)
            if self.can_name == "ZLG_USBCAN":
                send_len = DLL.VCI_Transmit(self.dev_type, self.dev_index, self.can_index, p_send_buf, frame_num)
            elif self.can_name == "GC_USBCAN":
                send_len = DLL.Transmit(self.dev_type, self.dev_index, self.can_index, p_send_buf, frame_num)

            if send_len == (frame_num):
                return send_len
            else:

                return STATUS_ERR
        else:
            return STATUS_ERR

    def read_err_info(self):
        err_info = VCI_ERR_INFO()
        p_err_info = P_VCI_ERR_INFO(err_info)
        if self.can_name == "ZLG_USBCAN":
            ret = DLL.VCI_ReadErrInfo(self.dev_type, self.dev_index, self.can_index, p_err_info)
        elif self.can_name == "GC_USBCAN":
            ret = DLL.ReadErrInfo(self.dev_type, self.dev_index, self.can_index, p_err_info)

        if ret == STATUS_OK:
            err_code = p_err_info.contents.ErrCode
            return err_code
        else:
            #logger.info("can not get the err_code")
            return STATUS_ERR


class USBCAN_2E_U(CanTunnel):
    def __init__(self, can_dev, can_index):
        super(USBCAN_2E_U, self).__init__(can_dev, can_index)

    def init_can(self, can_id):
        init_config = VCI_INIT_CONFIG(c_long(0), c_long(0), c_long(0), c_ubyte(0), c_ubyte(0), c_ubyte(0), c_ubyte(0))
        p_init_config = P_VCI_INIT_CONFIG(init_config)

        if not self.__set_baurdrate():
            #logger.info("set baurdrate err")
            return STATUS_ERR

        if self.can_name == "ZLG_USBCAN":
            ret = DLL.VCI_InitCAN(self.dev_type, self.dev_index, self.can_index, p_init_config)
        elif self.can_name == "GC_USBCAN":
            ret = DLL.InitCAN(self.dev_type, self.dev_index, self.can_index, p_init_config)


        if ret != STATUS_OK:

            return STATUS_ERR

        # if not self.__set_filters(can_id):
        #
        #     return STATUS_ERR

        if not self.__start_filters():

            return STATUS_ERR

    def __set_baurdrate(self):
        # 1000k
        pdata = c_long(c_long(0x060003))
        if self.can_name == "ZLG_USBCAN":
            ret = DLL.VCI_SetReference(self.dev_type, self.dev_index, self.can_index, SET_BAUDRATE_CMD, pdata)
        elif self.can_name == "GC_USBCAN":
            ret = DLL.VCI_SetReference(self.dev_type, self.dev_index, self.can_index, SET_BAUDRATE_CMD, pdata)

        return ret

    def __set_filters(self, can_id):
        filter_record = VCI_FILTER_RECORDER(0, c_long(can_id), c_long(can_id))
        p_filter_record = P_VCI_FILTER_RECORDER(filter_record)

        if self.can_name == "ZLG_USBCAN":
            ret = DLL.VCI_SetReference(self.dev_type, self.dev_index, self.can_index, SET_FILTER_CMD, p_filter_record)
        elif self.can_name == "GC_USBCAN":
            ret = DLL.VCI_SetReference(self.dev_type, self.dev_index, self.can_index, SET_FILTER_CMD, p_filter_record)

        return ret

    def __start_filters(self):

        if self.can_name == "ZLG_USBCAN":
            ret = DLL.VCI_SetReference(self.dev_type, self.dev_index, self.can_index, START_FILTER_CMD, c_void_p())
        elif self.can_name == "GC_USBCAN":
            ret = DLL.VCI_SetReference(self.dev_type, self.dev_index, self.can_index, START_FILTER_CMD, c_void_p())

        return ret


class USBCAN_II(CanTunnel):
    def __init__(self, can_dev, can_index):
        super(USBCAN_II, self).__init__(can_dev, can_index)

    def init_can(self):
        init_config = self.__get_init_cfg_by_id()
        p_init_config = P_VCI_INIT_CONFIG(init_config)

        if self.can_name == "ZLG_USBCAN":
            ret = DLL.VCI_InitCAN(self.dev_type, self.dev_index, self.can_index, p_init_config)
        elif self.can_name == "GC_USBCAN":
            ret = DLL.InitCAN(self.dev_type, self.dev_index, self.can_index, p_init_config)

        return ret

    # one id filter
    def __get_init_cfg_by_id(self):

        # receive all canid
        ACR = 0x00000000
        AMR = 0xFFFFFFFF
        reserved = 0

        # one id filter
        filter_byte = 1

        # can baudrate 1000K
        timing0 = 0x00
        timing1 = 0x14

        # normal mode
        mode = 0

        return VCI_INIT_CONFIG(ACR, AMR, reserved, filter_byte, timing0, timing1, mode)


class CanDev(object):
    def __init__(self, pdcan_name,dev_type,dev_index, can_index):
        self.name      = pdcan_name
        self.dev_type  = c_int(dev_type)
        self.dev_index = c_int(dev_index)
        self.__is_open = False

        self.select_can_dev(pdcan_name)

        reserved = c_int(0)

        for i in range(3):
            if pdcan_name == "ZLG_USBCAN":
                ret = DLL.VCI_OpenDevice(self.dev_type,self.dev_index , 0)
            elif pdcan_name == "GC_USBCAN":
                ret = DLL.OpenDevice(self.dev_type, self.dev_index, 0)

            if ret == STATUS_OK:
                self.__is_open = True
                break
        else:
            raise  sys_infor.CAN_OPEN_FAILED
            pass

        self.tunnel = self.get_can_dev(dev_type, can_index)

    @property
    def is_open(self):
        return self.__is_open

    def close(self):
        self.__is_open = False

        if self.name == "ZLG_USBCAN":
            return DLL.VCI_CloseDevice(self.dev_type, self.dev_index)
        elif self.name == "GC_USBCAN":
            return DLL.CloseDevice(self.dev_type, self.dev_index)

        return None


    def select_can_dev(self, name):
        global DLL
        if name == "ZLG_USBCAN":
            dll_path = "protocol/tool/ControlCAN.dll"
        elif name == "GC_USBCAN":
            dll_path = "protocol/tool/ECanVci64.dll"

        DLL = windll.LoadLibrary(str(dll_path))

    def get_can_dev(self, dev_type, can_index):

        if dev_type == dev_type:
            return USBCAN_II(self, can_index)

        elif dev_type == USBCAN_2E_U_TYPE:
            return USBCAN_2E_U(self, can_index)


class RecvCanData(threading.Thread):
    def __init__(self, can_tunnel):
        super(RecvCanData, self).__init__()
        self.stopped = False
        self.dev = can_tunnel

    def run(self):

        while not self.stopped:
            num = self.dev.get_receive_num()
            if num > 0:

                recv_data = self.dev.recv_data(num)
                for data in recv_data:
                    self.dev.recv_queue_handle.put(data)

            time.sleep(0.1)

    def stop(self):
        self.stopped = True

