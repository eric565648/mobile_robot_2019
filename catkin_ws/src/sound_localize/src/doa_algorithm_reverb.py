#!/usr/bin/env python
## Copyright 2019 PME Tommy,Cheng, Shinchen-Huang
##
## *****************************************************
##  DESCRIPTION :
##  socket of server  main.py
## use:
# please set ip  and host port first in the bottom accroding to environments
import time
import threading
import numpy as np
import os
import logging
import math
from utils import *
from scipy import signal
import socket
from tra_process_new_th import tra_process_new_th # for denoising

from ucaposition import ucaposition
from Mix3D_Pro_function_from_mic import Mix3D_Pro_function_from_mic
from MUSIC_location_theta_phi import MUSIC_location_theta_phi
from mwf_process_new_m2_th import mwf_process_new_m2_th

import rospy
from sound_localize.msg import SoundBearing, SoundRaw

M               = 5
r               = 0.05
Center_X        = 0
Center_Y        = 0
Center_Z        = 0
leadlag         = 15
SorNum          = 1
rate            = 16000

try:
    # python2 supporting
    import Queue
except:
    # python3
    import queue as Queue

class DOAHolder(object):
    """docstring for DOAHolder."""
    def __init__(self):
        super(DOAHolder, self).__init__()
        # ros init
        rospy.init_node('sound_doa',anonymous=False)

        # set up
        self.count = 0
        self.avgFrames=[]

        # total number of data = channels*frame = 2048*8 =16384
        # one data of chunk  = 2 bit
        # -> 16384*2 = 32768

        # #use one frame
        # while True:
        #     count = 0
        #     xx = []
        #     while count < 32 :
        #         data_temp = conn.recv(1024,socket.MSG_WAITALL)
        #         xx_temp = np.frombuffer(data_temp,'int16')
        #         xx = np.concatenate((xx,xx_temp))
        #         count +=1
        #     direction = micarray.beamforming(xx)
        #     conn.sendall(str(int(np.floor(direction))).encode('utf-8'))
        # conn.close()


        # Publisher
        self.pubDirection = rospy.Publisher('sound_direction', SoundBearing, queue_size=5)
        # Subscriber
        subSound = rospy.Subscriber('sound_raw', SoundRaw, self.sound_raw_cb, queue_size=2)

    def sound_raw_cb(self, sound_msg):

        global M, r, Center_X, Center_Y, Center_Z, leadlag, SorNum, rate

        time_len = 2

        self.count += 1
        self.count %= time_len
        if self.count==1:
            self.avgFrames = np.array(sound_msg.data).astype(float) / time_len
        else:
            self.avgFrames = self.avgFrames+np.array(sound_msg.data).astype(float)/time_len
            #avgFrames = np.concatenate((avgFrames,xx),axis=0)

        # print("avgFrame shape: ", self.avgFrames.shape)
        if self.count == 0:
            self.avgFrames = self.avgFrames / 2
            beam_start = rospy.Time.now()

            raw_sigs_f = self.avgFrames / 32768.0

            MicSignal = np.zeros((8,len(self.avgFrames[1::8])))
            # print("Mic len", MicSignal.shape)

            for i in range(8):
                MicSignal[i]=raw_sigs_f[i::8]
            # print(MicSignal)

            # giving information
            p_new           = np.zeros([M,2048],dtype = float)

            for i in range(M):
                # p_new[i,:]  = p[i,:]
                # p_new[i,:] = tra_process_new_th(MicSignal[i,:].T)
                p_new[i,:] = mwf_process_new_m2_th(MicSignal[i,:].T,MicSignal[i+1,:].T,rate)
            # print(p_new)
            # print(p_new)
            micpos          = ucaposition(M,r,Center_X,Center_Y,Center_Z,leadlag)
            P_half          = Mix3D_Pro_function_from_mic(p_new)
            # print(P_half)
            Location        = MUSIC_location_theta_phi(SorNum,micpos,P_half)

            print('azimuth:', Location[0,0])
            print('elevation:', Location[0,1])
            azimuth = Location[0,0]
            elevation = Location[0, 1]

            beam_end = rospy.Time.now()
            print("beam time: ", (beam_end-beam_start).to_sec())
            print("==========================")
            # conn.sendall(str(int(np.floor(direction))).encode('utf-8'))
            # conn.sendall(str(int(np.floor(azimuth))).encode('utf-8'))
            self.avgFrames = []

            #ros publisher
            soundbearing = SoundBearing()
            soundbearing.header.stamp = sound_msg.header.stamp
            soundbearing.azimuth = azimuth
            soundbearing.elevation = elevation
            soundbearing.obj_id = 1
            self.pubDirection.publish(soundbearing)

if __name__ == '__main__':

    d = DOAHolder()
    rospy.spin()
