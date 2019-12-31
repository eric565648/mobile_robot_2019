#!/usr/bin/env python

# This node vote collect bearing of sound over a time
# and vote for a most appear bearing
# just like duckietown classic lane filter

import rospy
import numpy as np
import math
from sound_localize.msg import SoundBearing
from std_msgs.msg import Bool

class BearingFilter(object):
    """docstring for BearingFilter."""
    def __init__(self):
        super(BearingFilter, self).__init__()

        # parameters
        self.filter_data_threshold = rospy.get_param("~filter_data_threshold", 20) # how many data used in a vote
        self.switch = False # if we can collect the bearing or not (collect while stop)
        self.switch_delay = 3 # wait for delay after active switch

        # variables
        self.azi_min = 0
        self.ele_min = 0
        self.azi_delta = 5
        self.ele_delta = 5
        self.azi_digi = 360/self.azi_delta
        self.ele_digi = 90/self.ele_delta
        self.filter_azi = np.zeros((self.azi_digi)) # azimuth angle every 5 degree
        self.filter_ele = np.zeros((self.ele_digi)) # elevation angle every 5 degree
        self.last_stamp = None

        # Publisher
        self.bearing_pub = rospy.Publisher("filtered_bearing", SoundBearing, queue_size = 3)

        ## msg filter sync range and bearing images
        bearing_sub = rospy.Subscriber('sound_direction', SoundBearing, self.bearing_cb, queue_size=5)
        switch_sub = rospy.Subscriber('filter_switch', Bool, self.switch_cb, queue_size=1)

    def bearing_cb(self, bearing_msg):

        if not self.switch: # clear the vote and return if not detecting
            self.filter_azi = np.zeros((self.azi_digi))
            self.filter_ele = np.zeros((self.ele_digi))
            return

        azi_i = int(math.floor((bearing_msg.azimuth - self.azi_min)/self.azi_delta))
        ele_i = int(math.floor((bearing_msg.elevation - self.ele_min)/self.ele_delta))

        self.filter_azi[azi_i] += 1
        self.filter_ele[ele_i] += 1

        self.last_stamp = bearing_msg.header.stamp

        print "Data amount: ", np.sum(self.filter_azi)
        if np.sum(self.filter_azi) >= self.filter_data_threshold and self.filter_data_threshold != 0:
            azi_max = np.argmax(self.filter_azi)
            ele_max = np.argmax(self.filter_ele)

            bearing_msg_pub = SoundBearing()
            bearing_msg_pub.azimuth = azi_max * self.azi_delta + self.azi_min
            bearing_msg_pub.elevation = ele_max * self.ele_delta + self.ele_min
            bearing_msg_pub.header.stamp = self.last_stamp
            self.bearing_pub.publish(bearing_msg_pub)

            print "azimuth: ", bearing_msg_pub.azimuth
            print "elevation: ", bearing_msg_pub.elevation


    def switch_cb(self, switch_msg):

        # if threshold == 0
        # publish result while end this point
        if self.filter_data_threshold == 0 and self.switch == True and switch_msg.data == False:
            azi_max = np.argmax(self.filter_azi)
            ele_max = np.argmax(self.filter_ele)

            bearing_msg_pub = SoundBearing()
            bearing_msg_pub.azimuth = azi_max * self.azi_delta + self.azi_min
            bearing_msg_pub.elevation = ele_max * self.ele_delta + self.ele_min
            bearing_msg_pub.header.stamp = self.last_stamp
            self.bearing_pub.publish(bearing_msg_pub)

        self.switch = switch_msg.data
        if self.switch == False:
            self.filter_azi = np.zeros((self.azi_digi))
            self.filter_ele = np.zeros((self.ele_digi))

if __name__ == '__main__':
	rospy.init_node('bearing_filter',anonymous=False)
	bf = BearingFilter()
	rospy.spin()
