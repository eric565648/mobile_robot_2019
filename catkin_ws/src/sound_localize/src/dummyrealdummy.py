#!/usr/bin/env python

import rospy
import numpy as np
import math
from sound_localize.msg import SoundBearing
from pozyx_ros.msg import DeviceRangeArray
from anchor_measure.msg import PoseDirectional # pose with direction
from geometry_msgs.msg import PoseArray, Pose
import tf

import message_filters

class Dummy(object):
    """docstring for Dummy."""
    def __init__(self):
        super(Dummy, self).__init__()

        # variables

        self.sw_pub = rospy.Publisher("pozyx_range_array", DeviceRangeArray, queue_size = 5)

        ## msg filter sync rgb and d images
        sub_p = rospy.Subscriber('pozyx_range', DeviceRangeArray, self.sw_cb, queue_size = 5)

    def sw_cb(self, range_msg):

        range_msg.header.stamp = range_msg.rangeArray[0].header.stamp
        self.sw_pub.publish(range_msg)


if __name__ == '__main__':
	rospy.init_node('dummt',anonymous=False)
	d = Dummy()
	rospy.spin()
