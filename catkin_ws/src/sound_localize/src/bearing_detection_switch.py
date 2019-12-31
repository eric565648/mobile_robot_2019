#!/usr/bin/env python

# This node vote collect bearing of sound over a time
# and vote for a most appear bearing
# just like duckietown classic lane filter

import rospy
import numpy as np
import math
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

class BearingSwitch(object):
    """docstring for BearingSwitch."""
    def __init__(self):
        super(BearingSwitch, self).__init__()

        self.last_button = True

        # Publisher
        self.switch_pub = rospy.Publisher("filter_switch", Bool, queue_size = 1)
        # Subscriber
        joy_sub = rospy.Subscriber('/joy_teleop/joy', Joy, self.joy_cb, queue_size=1)

    def joy_cb(self, switch_msg):

        bool_msg = Bool()

        if switch_msg.buttons[0] == 0:
            bool_msg.data = True
        else:
            bool_msg.data = False

        self.switch_pub.publish(bool_msg)
        # print "Switch: ", bool_msg.data

if __name__ == '__main__':
	rospy.init_node('bearing_detection_switch',anonymous=False)
	bs = BearingSwitch()
	rospy.spin()
