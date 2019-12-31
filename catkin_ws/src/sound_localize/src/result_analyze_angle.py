#!/usr/bin/env python

import numpy as np
import rospy
import rospkg
from tf import transformations as tr
from sound_localize.msg import SoundBearing
from std_msgs.msg import String
import math
from visualization_msgs.msg import Marker
import yaml

np.set_printoptions(suppress=True)

class SWLandmark(object):
    """docstring for SWLandmark."""
    def __init__(self):
        super(SWLandmark, self).__init__()

        self.now_state = None

        ## subscribers
        string_sub = rospy.Subscriber('state', String, self.state_cb, queue_size=1)
        pozyx_sub = rospy.Subscriber('sound_direction', SoundBearing, self.sound_cb,queue_size=1)

        ## datas
        self.bearings = {}

    def state_cb(self, state_msg):
        self.now_state = state_msg.data

        # angle = (ord(now_state[0])-98)*30
        # distance = now_state[1:3]

        if self.now_state == 'end':
            self.dump_data()

    def sound_cb(self, sound_msg):
        if self.now_state == None:
            return

        if self.now_state not in self.bearings:
            self.bearings[self.now_state] = np.array([sound_msg.azimuth, sound_msg.elevation])
        else:
            self.bearings[self.now_state] = np.vstack((self.bearings[self.now_state], [sound_msg.azimuth, sound_msg.elevation]))

    def dump_data(self):
        print("Save to file.")
        with open(rospkg.RosPack().get_path('sound_localize')+"/config/1225_be.yaml", 'w') as outfile:
            yaml.dump(self.bearings, outfile, default_flow_style=False)

if __name__ == '__main__':
	rospy.init_node('result_analyze',anonymous=False)
	s = SWLandmark()
	rospy.spin()
