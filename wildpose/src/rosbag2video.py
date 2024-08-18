#!/usr/bin/python3
import os
import sys
import argparse
from pprint import pprint
import queue
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from tqdm import tqdm
import shlex, subprocess
import ffmpeg

import sqlite3
from rosidl_runtime_py.utilities import get_message
import rclpy
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message

MJPEG_VIDEO = 1
RAWIMAGE_VIDEO = 2
VIDEO_CONVERTER_TO_USE = "ffmpeg"   # or you may want to use "avconv"


parser = argparse.ArgumentParser()
parser.add_argument('--input_db', type=str, required=True, help='The input rosbag file [db3].')
parser.add_argument('--topic', type=str, default='/image_raw', help='Only the images from topic "topic" are used for the video output.')
args = parser.parse_args()


class BagFileParser():
    # https://answers.ros.org/question/358686/how-to-read-a-bag-file-in-ros2/
    def __init__(self, bag_file):
        self.conn = sqlite3.connect(bag_file)
        self.cursor = self.conn.cursor()

        ## create a message type map
        topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        self.topic_type = {name_of: type_of for id_of, name_of,type_of in topics_data}
        self.topic_id = {name_of: id_of for id_of,name_of, type_of in topics_data}
        self.topic_msg_message = {name_of: get_message(type_of) for id_of, name_of,type_of in topics_data}

    def __del__(self):
        self.conn.close()

    def get_messages(
        self,
        topic_name: str,
        limit: int = None,
    ):
        '''
        Return [(timestamp0, message0), (timestamp1, message1), ...]
        '''
        topic_id = self.topic_id[topic_name]

        query = f"SELECT timestamp, data FROM messages WHERE topic_id = {topic_id} "
        if limit is not None:
            query += f"LIMIT {limit} "

        rows = self.cursor.execute(query).fetchall()
        # Deserialise all and timestamp them
        return [
            (timestamp, deserialize_message(data, self.topic_msg_message[topic_name]))
            for timestamp,data in rows
        ]


def header2timestamp(header):
    result = int('{}{:09d}'.format(header.stamp.sec, header.stamp.nanosec))
    return result

def header2imgcnt(header):
    result = int(header.frame_id[2:])
    return result


def rosbag2video(
    filename: str,
    topic: str = "",
    output_filename: str = "",
    opt_out_file: str = "",
    opt_prefix: str = None,
    verbose: bool = False
):
    if verbose:
        print("Bagfile: {}".format(filename))

    if opt_prefix is None:
        # create the output in the same folder and name as the bag file minu '.bag'
        opt_prefix = filename[:-4]

    # Go through the bag file
    bag = BagFileParser(filename)
    if verbose:
        print("Bag opened.")

    # output file path
    if opt_out_file == "":
        out_file = opt_prefix + str(topic).replace("/", "_")+".mp4"
    else:
        out_file = opt_out_file
    if verbose:
        print(f"Using output file {out_file} for topic {topic}.")

    # get the real camera parameters
    messages = bag.get_messages('/xi_image_info')
    img_info = messages[0][1]   # the info is basically same over all messages

    # get /image_raw messages
    topic = args.topic
    messages = bag.get_messages(topic)
    messages = sorted(  # sort the messages by the timestamp
        messages,
        key=lambda x: header2timestamp(x[1].header)
    )
    for _, msg in messages:
        print(msg.header.frame_id)

    # get FPS
    video_duration_ns = header2timestamp(messages[-1][1].header) - header2timestamp(messages[0][1].header)
    n_frames = header2imgcnt(messages[-1][1].header) - header2imgcnt(messages[0][1].header)
    fps = int(n_frames / (video_duration_ns * 1e-9))
    if verbose:
        print(f'FPS: {fps}')

    # open video file
    width = int(img_info.width)
    height = int(img_info.height)
    size = (width, height)
    gst_command = f'appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! nvv4l2h264enc ! h264parse ! qtmux name=mux ! filesink location={out_file}'
    video = cv2.VideoWriter(gst_command, cv2.CAP_GSTREAMER, 0, fps, size)
    if not video.isOpened():
        print("ERROR: Failed to open output")
        exit(1)

    # rendering
    bridge = CvBridge()
    t_first = header2timestamp(messages[0][1].header) * 1e-9
    t_video = 0
    img_cnt = 0
    for _, msg in tqdm(messages):
        t_ns = header2timestamp(msg.header) * 1e-9  # timestamp (nanosec)
        t_file = t_ns - t_first

        while t_video <= t_file:
            try:
                if msg.format.find("jpeg") != -1:
                    if msg.format.find("8")!=-1 and (msg.format.find("rgb")!=-1 or msg.format.find("bgr")!=-1 or msg.format.find("bgra")!=-1):
                        np_arr = np.fromstring(msg.data, np.uint8)
                        cv_image = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
                        video.write(cv_image)
                    elif msg.format.find("mono8")!=-1:
                        np_arr = np.fromstring(msg.data, np.uint8)
                        cv_image = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
                        video.write(cv_image)
                    elif msg.format.find("16UC1")!=-1:
                        np_arr = np.fromstring(msg.data, np.uint16)
                        cv_image = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
                        video.write(cv_image)
                    else:
                        print(f'unsupported jpeg format: {msg.format}.{topic}')
            # has no attribute 'format'
            except AttributeError:
                try:
                    pix_fmt = None
                    if msg.encoding.find("mono8")!=-1 or msg.encoding.find("8UC1")!=-1:
                        pix_fmt = "gray"
                        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                    elif msg.encoding.find("bgra")!=-1:
                        pix_fmt = "bgra"
                        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                    elif msg.encoding.find("bgr8")!=-1:
                        pix_fmt = "bgr24"
                        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                    elif msg.encoding.find("bggr8")!=-1:
                        pix_fmt = "bayer_bggr8"
                        cv_image = bridge.imgmsg_to_cv2(msg, "bayer_bggr8")
                    elif msg.encoding.find("rggb8")!=-1:
                        pix_fmt = "bayer_rggb8"
                        cv_image = bridge.imgmsg_to_cv2(msg, "bayer_rggb8")
                    elif msg.encoding.find("rgb8")!=-1:
                        pix_fmt = "rgb24"
                        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                    elif msg.encoding.find("16UC1")!=-1:
                        pix_fmt = "gray16le"
                    else:
                        print('unsupported encoding:', msg.encoding, topic)
                        #exit(1)
                    if pix_fmt is not None:
                        video.write(cv_image)
                        # cv2.imwrite(opt_prefix + f'{img_cnt}.jpg', cv_image)
                        img_cnt += 1
                except AttributeError:
                    # maybe theora packet
                    # theora not supported
                    if verbose:
                        print("Could not handle this format. Maybe thoera packet? theora is not supported.")
            t_video += 1.0 / fps
    # closing
    video.release()


if __name__ == "__main__":
    # bagfile = BagFileParser(args.input_db)

    # image_msgs = bagfile.get_messages('/image_raw', limit=1)

    # img = image_msgs[0][1]  # sensor_msgs.msg._image.Image
    # timestamp = img.header.stamp
    # print(type(image_msgs[0][0]), image_msgs[0][0])
    # print(type(timestamp.sec), timestamp)
    # print(img.encoding)

    rosbag2video(args.input_db, args.topic, verbose=True)