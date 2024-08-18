#!/usr/bin/python3
import rclpy
from rclpy.node import Node

import can
import struct
from protocol.sdk import CmdCombine


BITRATE = 1000000
Seq_Init_Data = 0x0002


def command_generator(cmd_type: str, cmd_set: str, cmd_id: str, data: str) -> bytearray:
    dji_cmd = CmdCombine.combine(cmd_type=cmd_type, cmd_set=cmd_set, cmd_id=cmd_id, data=data)
    return bytearray([int(b, 16) for b in dji_cmd.split(':')])


class DjiRs3Node(Node):
    def __init__(self):
        super().__init__("dji_rs3_node")

        # Node parameters
        self.declare_parameter("channel", "can0")
        self.channel_ = self.get_parameter("channel").value

        self.bus_ = can.interface.Bus(
            bustype='socketcan',
            channel=self.channel_,
            bitrate=BITRATE
        )
        self.send_id_ = 0x223
        self.rev_id_ = 0x222

        self.timer_ = self.create_timer(1.0, self.send_data)

        self.get_logger().info("dji_rs3_node started.")

    def get_data(self):
        msg = self.bus_.recv()
        if msg is not None:
            self.get_logger().info(f'{msg}')
            self.get_logger().info(f'{type(msg)}\t{msg.dlc}\t{msg.data}')

    def send_data(self):
        # hex_data = struct.pack(
        #     '<3h2B',
        #     0, # yaw,
        #     0, # roll,
        #     90 * 10, # pitch,
        #     0x01, # ctrl_byte,
        #     0x14, # time_for_action
        # )
        # pack_data = ['{:02X}'.format(i) for i in hex_data]
        # cmd_data = ':'.join(pack_data)
        # cmd = CmdCombine.combine(cmd_type='03', cmd_set='0E', cmd_id='00', data=cmd_data)
        # self.get_logger().info(f'cmd: {cmd}')


        # $ cansend can0 223#AA1A000300000000 & \
        # cansend can0 223#2211A2420E002000 & \
        # cansend can0 223#3000400001147B40 & \
        # cansend can0 223#97BE
        # msg = can.Message(
        #     arbitration_id=self.send_id_,
        #     is_extended_id=False,
        #     is_rx=False,
        #     data=bytearray([
        #         0xAA, # SOF
        #         0x1A, 0x00, # Ver/Length
        #         0x03, # CmdType
        #         0x00, # ENC
        #         0x00, 0x00, 0x00, # RES
        #         0x22, 0x11, # SEQ
        #         0xA2, 0x42, # CRC-16
        #         0x0E, 0x00, 0x20, 0x00, 0x30, 0x00, 0x40, 0x00, 0x01, 0x14, # DATA
        #         0x7B, 0x40, 0x97, 0xBE # CRC-32
        #     ])
        # )

        hex_data = struct.pack(
            '<3h2B',    # format: https://docs.python.org/3/library/struct.html#format-strings
            0, # yaw,
            0, # roll,
            90 * 10, # pitch,
            0x01, # ctrl_byte,
            0x14, # time_for_action
        )
        pack_data = ['{:02X}'.format(i) for i in hex_data]
        cmd_data = ':'.join(pack_data)
        send_data = command_generator(
            cmd_type='03',
            cmd_set='0E',
            cmd_id='00',
            data=cmd_data
        )
        msg = can.Message(
            arbitration_id=self.send_id_,
            is_extended_id=False,
            is_rx=False,
            data=send_data
        )

        self.get_logger().info(f'msg: {msg}')
        self.bus_.send(msg, timeout=0.5)


def main(args=None):
    rclpy.init(args=args)
    node = DjiRs3Node()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()