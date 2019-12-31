#!/usr/bin/env python

import numpy as np
import rospy
import rospkg
from tf import transformations as tr
from sound_localize.msg import SoundRaw
from std_msgs.msg import String
import math

class SoundSave(object):
    """docstring for SoundSave."""
    def __init__(self):
        super(SoundSave, self).__init__()

        self.filename = rospkg.RosPack().get_path('sound_localize')+"/config/0_4/"

        self.frame = 0

        ## subscribers
        pozyx_sub = rospy.Subscriber('sound_raw', SoundRaw, self.sound_cb,queue_size=1)

    def sound_cb(self, sound_msg):

        chunk = np.array(sound_msg.data)
        print("chunk len", chunk.shape)

        MicSignal = np.zeros((8,len(chunk[1::8])))
        print("Mic len", MicSignal.shape)

        for i in range(8):
            MicSignal[i]=chunk[i::8]

        self.dump_data(MicSignal, self.frame)
        self.frame += 1

    def dump_data(self, array,frame):
        print("Save to file.")

        for i in range(len(array)):
            filename = self.filename + 'frame_' + str(frame) + '_channel_' + str(i+1) + '.npy'
            np.save(filename, array[i])

if __name__ == '__main__':
	rospy.init_node('sound_save',anonymous=False)
	s = SoundSave()
	rospy.spin()
