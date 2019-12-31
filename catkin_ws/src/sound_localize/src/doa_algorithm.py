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

import rospy
from sound_localize.msg import SoundBearing, SoundRaw

try:
    # python2 supporting
    import Queue
except:
    # python3
    import queue as Queue

logger = logging.getLogger('MicArray')


class MicArray(object):

    SOUND_SPEED = 343.0
    """
    UCA (Uniform Circular Array)

    Design Based on Respeaker 7 mics array architecture
    """
    def __init__(self, fs=16000, nframes=2048, radius=0.046, num_mics=6
                    , quit_event=None, name='respeaker-7'):
 ######################################################################### please don't modify below
        self.radius     = radius
        self.fs         = fs
        self.nframes    = nframes
        self.nchannels  = (num_mics + 2 if name == 'respeaker-7' else num_mic)
        self.num_mics   = num_mics
        self.max_delay  = radius * 2 / MicArray.SOUND_SPEED

        self.quit_event = quit_event if quit_event else threading.Event()

        # multi-channels input
        self.listen_queue = Queue.Queue()

        # build tdoa matrix
        mic_theta           = np.arange(0, 2*np.pi, 2*np.pi/6)
        self.tdoa_matrix    = np.array([np.cos(mic_theta[1:]), np.sin(mic_theta[1:])]).T

        self.tdoa_matrix -= np.ones((len(mic_theta)-1, 2)) \
                             * np.array([np.cos(mic_theta[0]), np.sin(mic_theta[0])])
#################### wired #######################################
        self.tdoa_measures  = np.ones(((len(mic_theta)-1, ))) \
                                * MicArray.SOUND_SPEED / (self.radius)       # diameter => radius
#################### wired #######################################
        self.MicPos = self.radius*np.array([np.cos(mic_theta),np.sin(mic_theta),np.zeros(6)])
#############################################

    def MUSIC_implement(self,buff):
        '''
        use:P_half = Mix_from_mic(MicPos,buff)
        arguments:
        P_half : frequency domain signal
        MicPos : position of Microphone array
        buff : Microphone signal gained from socket
        '''
        P_half = Mix_from_mic(self.MicPos,buff) # translate time domain into frequency domain

        # four narrowband  frequency index
        #select_range1 =np.arange(304,324,1)
        # select_range1 =np.arange(250,350,1)
        select_range1 =np.arange(384,404,1)
        #select_range1 =np.arange(407,427,1)
        # select_range1 =np.arange(430,450,1)
        # select_range1 =np.arange(361,381,1)()
        # select_range_speech =np.arange(256,384,1)
        '''
        use:azimuth_angle,elevation_angle = MUSIC_Localization_freqrange(P_half,MicPos,SorNum,select_range)
        arguments:
        P_half : frequency domain signal
        MicPos : position of Microphone array
        SorNum : number of sound source (must less than number of microphone)
        select_range: the range of chosen frequency
        '''
        # azimuth_angle,elevation_angle = MUSIC_Localization_freqrange(P_half,self.MicPos,1,select_range1)
        # azimuth_angle_only=MUSIC_Localization_freqrange_theta_given_phi(P_half,self.MicPos,1,select_range1,15)
        azimuth_angle,elevation_angle=MUSIC_Localization_freqrange_grid_search(P_half,self.MicPos,1,select_range1)
        # assume elevation_angle = 20
        # azimuth_angle,elevation_angle = MUSIC_Localization_freqrange_theta(P_half,self.MicPos,1,select_range1)

        # azimuth_angle,azimuth_angle2= Multi_MUSIC_Localization_freqrange_theta(P_half,self.MicPos,2,select_range_speech)

        '''
        use:azimuth_angle,elevation_angle = MUSIC_Localization(P_half,MicPos,SorNum)
        arguments:
        P_half : frequency domain signal
        MicPos : position of Microphone array
        SorNum : number of sound source (must less than number of microphone)
        '''
        #azimuth_angle,elevation_angle=MUSIC_Localization(P_half,self.MicPos,1)


        '''
        use:azimuth_angle,elevation_angle = MUSIC_PSO_localization(P_half,MicPos,SorNum) fast MUSIC
        arguments:
        P_half : frequency domain signal
        MicPos : position of Microphone array
        SorNum : number of sound source (must less than number of microphone)
        '''
        #azimuth_angle,elevation_angle=MUSIC_PSO_localization(P_half,self.MicPos,2)

        return azimuth_angle,elevation_angle

    def beamforming(self, chunk):
        # decode from binary stream
        # casting int16 to double floating number
        chunk =np.divide( chunk ,(2**15))
        #########################################

        #initialize microphone signal matrix use first 6
        MicSignal = np.zeros((6,len(chunk[1::8])))
        print("Mic len", MicSignal.shape)

        for i in range(6):
            MicSignal[i]=chunk[i::8]
            MicSignal[i] = tra_process_new_th(MicSignal[i])


        buff =np.divide(MicSignal,np.max(np.max(np.abs(MicSignal),axis=1)))

        #np.save('micsignal2l',MicSignal)

	'''
        #### use TDOA
        MIC_GROUP_N = self.num_mics - 1
        MIC_GROUP = [[i, 0] for i in range(1, MIC_GROUP_N+1)]
        tau = [0] * 5

        direction = DOA(MicSignal,self.max_delay,MIC_GROUP,self.tdoa_matrix,self.tdoa_measures,self.fs)

        # print('@ {:.2f} @ {:.2f}'.format(direction[0], direction[1]),end='\n')
        return direction[0], direction[1]
	'''


        ###MUSIC
        azimuth_angle, elevation_angle =self.MUSIC_implement(buff)
        #azimuth_angle, azimuth_angle2 =self.MUSIC_implement(buff)
        # print('@ {:.2f} @{:.2f}'.format(azimuth_angle,elevation_angle))
        return azimuth_angle, elevation_angle


class DOAHolder(object):
    """docstring for DOAHolder."""
    def __init__(self):
        super(DOAHolder, self).__init__()
        # ros init
        rospy.init_node('sound_doa',anonymous=False)

        logging.basicConfig(level=logging.DEBUG)

        # MicArray
        self.micarray = MicArray(fs=16000, nframes=2048, radius=0.046, num_mics=6,quit_event=None, name='respeaker-7')

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

        time_len = 2

        self.count += 1
        self.count %= time_len
        if self.count==1:
            self.avgFrames = np.array(sound_msg.data).astype(float)
        else:
            self.avgFrames = np.append(self.avgFrames, np.array(sound_msg.data).astype(float))
            #avgFrames = np.concatenate((avgFrames,xx),axis=0)
        print("avgFrame shape: ", self.avgFrames.shape)
        if self.count == 0:
            self.avgFrames = self.avgFrames.astype(int)
            beam_start = rospy.Time.now()
            azimuth, elevation = self.micarray.beamforming(self.avgFrames)
            print("azimuth, elevation: ", azimuth, elevation)
            beam_end = rospy.Time.now()
            print("beam time: ", (beam_end-beam_start).to_sec())
            print("==========================")
            # conn.sendall(str(int(np.floor(direction))).encode('utf-8'))
            # conn.sendall(str(int(np.floor(azimuth))).encode('utf-8'))
            self.avgFrames = None

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
