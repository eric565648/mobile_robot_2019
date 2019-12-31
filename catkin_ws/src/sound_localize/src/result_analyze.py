#!/usr/bin/env python

import numpy as np
import rospy
import rospkg
from tf import transformations as tr
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from apriltags2_ros.msg import AprilTagDetection, AprilTagDetectionArray
from sensor_msgs.msg import Joy
from pozyx_ros.msg import DeviceRangeArray
from sound_localize.msg import SoundBearing
import math
from visualization_msgs.msg import Marker
import yaml

np.set_printoptions(suppress=True)

class SWLandmark(object):
    """docstring for SWLandmark."""
    def __init__(self):
        super(SWLandmark, self).__init__()

        ## Publisher
        self.marker_pub = rospy.Publisher("robot_visualize", Marker, queue_size =10)
        ## subscribers
        at_sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.apriltags_cb,queue_size=10)
        joy_sub = rospy.Subscriber('/joy_teleop/joy', Joy, self.joy_cb,queue_size=1)
        pozyx_sub = rospy.Subscriber('/pozyx_range', DeviceRangeArray, self.range_cb,queue_size=1)
        pozyx_sub = rospy.Subscriber('/sound_direction', SoundBearing, self.sound_cb,queue_size=1)

        ## To know blocks
        self.joy_lock = True
        self.joy_index = 0
        self.joy_time = None
        self.last_joy = None
        self.stop = False

        ## datas
        self.ranges = {}
        self.bearings = {}
        self.tags = {}

        # Marker
        vel = Marker()
        vel.header.frame_id = "map"
        vel.header.stamp = rospy.Time.now()
        vel.ns = "robot"
        vel.action = Marker.ADD
        vel.type = Marker.SPHERE
        vel.id = 0
        vel.scale.x = 0.1
        vel.scale.y = 0.1
        vel.scale.z = 0.1
        vel.color.a = 1.0
        vel.color.g = 1
        vel.color.b = 0
        self.robot_marker = vel

    def joy_cb(self, joy_msg):
        if self.stop:
            return

        # Wait 5 sec before start collecting data
        if self.joy_index == 0:
            rospy.sleep(5)
            self.joy_index += 1
            self.joy_time = joy_msg.header.stamp + rospy.Duration(5)
            self.last_joy = joy_msg.buttons[0]
            self.joy_lock = False
            print("joy index: ", self.joy_index)
            return
        if self.joy_lock and self.joy_index == 11: # no more than 11
            with open(rospkg.RosPack().get_path('sound_localize')+"/config/1127.yaml", 'w') as outfile:
                data = {}
                data['ranges'] = self.ranges
                data['bearings'] = self.bearings
                yaml.dump(data, outfile, default_flow_style=False)
            self.stop = True
            return

        change = joy_msg.buttons[0] - self.last_joy
        self.last_joy = joy_msg.buttons[0]

        if self.joy_lock and change==-1:
            rospy.sleep(2)
            self.joy_index += 1
            self.joy_time = joy_msg.header.stamp + rospy.Duration(2)
            self.joy_lock = False
            print("joy index: ", self.joy_index)
            return

        if not self.joy_lock and (joy_msg.header.stamp-self.joy_time).to_sec() > 12:
            print("joy lock")
            self.joy_lock = True


    def apriltags_cb(self, detections_msg):

        if self.joy_lock:
            return

        self.robot_marker.header.stamp = detections_msg.header.stamp
        for detection in detections_msg.detections:
            self.robot_marker.header.stamp = rospy.Time.now()
            x = detection.pose.pose.pose.position.z
            y = -detection.pose.pose.pose.position.x
            z = -detection.pose.pose.pose.position.y
            self.robot_marker.pose.position.x = -x
            self.robot_marker.pose.position.y = -y
            self.robot_marker.pose.position.z = -z

            self.marker_pub.publish(self.robot_marker)

    def range_cb(self, range_msg):
        if self.joy_lock:
            return

        if len(range_msg.rangeArray) == 0:
            return

        r = range_msg.rangeArray[0].distance/1000.

        if self.joy_index not in self.ranges:
            self.ranges[self.joy_index] = np.array([r])
        else:
            self.ranges[self.joy_index] = np.append(self.ranges[self.joy_index], r)

    def sound_cb(self, sound_msg):
        if self.joy_lock:
            return

        if self.joy_index not in self.bearings:
            self.bearings[self.joy_index] = np.array([sound_msg.azimuth, sound_msg.elevation])
        else:
            self.bearings[self.joy_index] = np.vstack((self.bearings[self.joy_index], np.array([sound_msg.azimuth, sound_msg.elevation])))

if __name__ == '__main__':
	rospy.init_node('result_analyze',anonymous=False)
	s = SWLandmark()
	rospy.spin()
