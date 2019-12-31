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

import rospy
from sound_localize.msg import SoundBearing
from std_msgs.msg import Int16MultiArray

try:
    # python2 supporting
    import Queue
except:
    # python3
    import queue as Queue

logger              = logging.getLogger('MicArray')

chunk_buffer = None
chunk_lock = True
azimuth_global = 0
chunk_time = None

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
        #select_range1 =np.arange(384,404,1)
        #select_range2 =np.arange(407,427,1)
        select_range4 =np.arange(430,450,1)
        #select_range4 =np.arange(361,381,1)
        #select_range_speech =np.arange(256,384,1)
        '''
        use:azimuth_angle,elevation_angle = MUSIC_Localization_freqrange(P_half,MicPos,SorNum,select_range)
        arguments:
        P_half : frequency domain signal
        MicPos : position of Microphone array
        SorNum : number of sound source (must less than number of microphone)
        select_range: the range of chosen frequency
        '''
        azimuth_angle,elevation_angle = MUSIC_Localization_freqrange(P_half,self.MicPos,1,select_range4)

        # assume elevation_angle = 20
        # azimuth_angle,elevation_angle = MUSIC_Localization_freqrange_theta(P_half,self.MicPos,1,select_range4)

        #azimuth_angle,azimuth_angle2= Multi_MUSIC_Localization_freqrange_theta(P_half,self.MicPos,2,select_range_speech)

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

        for i in range(6):
            MicSignal[i]=chunk[i::8]


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

def task(quit_event):
    # DEBUG
    global chunk_lock, chunk_buffer, azimuth_global
    micarray = MicArray(fs=16000, nframes=2048, radius=0.046, num_mics=6,quit_event=quit_event, name='respeaker-7')
    # conn, addr = s.accept()
    # print ('Connected by:', addr)
    count = 0
    avgFrames=[]
    '''
    total number of data = channels*frame = 2048*8 =16384
    one data of chunk  = 2 bytes
    -> 16384*2 = 32768
    '''
    # recieve data


    '''
    #use one frame
    while True:
        count = 0
        xx = []
        while count < 32 :
            data_temp = conn.recv(1024,socket.MSG_WAITALL)
            xx_temp = np.frombuffer(data_temp,'int16')
            xx = np.concatenate((xx,xx_temp))
            count +=1
        direction = micarray.beamforming(xx)
        conn.sendall(str(int(np.floor(direction))).encode('utf-8'))
    conn.close()
    '''
    pubDirection = rospy.Publisher('sound_direction', SoundBearing, queue_size=5)

    while not rospy.is_shutdown():
        #use two frame
        # print("music")

        #data = conn.recv(32768,socket.MSG_WAITALL)
        while chunk_lock:
            pass
        data = chunk_buffer
        chunk_lock = True
        count += 1
        count %= 2
        xx = np.frombuffer(data,'int16')
        if count==1:
            avgFrames = xx
        else:
            avgFrames = avgFrames+xx
            #avgFrames = np.concatenate((avgFrames,xx),axis=0)

        if count == 0:
            avgFrames =avgFrames / 2
            beam_start = rospy.Time.now()
            azimuth, elevation = micarray.beamforming(avgFrames)
            beam_end = rospy.Time.now()
            print("beam time: ", (beam_end-beam_start).to_sec())
            print("==========================")
            # conn.sendall(str(int(np.floor(direction))).encode('utf-8'))
            # conn.sendall(str(int(np.floor(azimuth))).encode('utf-8'))
            azimuth_global = azimuth
            avgFrames = None

            #ros publisher
            soundbearing = SoundBearing()
            soundbearing.header.stamp = chunk_time
            soundbearing.azimuth = azimuth
            soundbearing.elevation = elevation
            soundbearing.obj_id = 1
            pubDirection.publish(soundbearing)
    # conn.close()

    #s.close()

def rec_thread(quit_event):
    global chunk_lock, chunk_buffer, azimuth_global
    conn, addr = s.accept()

    pubFrameRaw = rospy.Publisher('sound_raw', Int16MultiArray, queue_size=1)

    while not rospy.is_shutdown():
        # con_start = rospy.Time.now()
        # It only takes 0.00011 secs to transfer
        # 32768 bits from respeaker to robots through Ethernet
        chunk_buffer = conn.recv(32768,socket.MSG_WAITALL)
        # con_end = rospy.Time.now()
        chunk_time = rospy.Time.now()
        # print("Receive")
        # print("con time: ", (con_end-con_start).to_sec())
        # print("==========================")

        xx = np.frombuffer(chunk_buffer,'int16')
        # print("The xx: ", xx)
        # print("xx len: ", xx.size)
        s_raw_msg = Int16MultiArray()
        s_raw_msg.data = xx
        pubFrameRaw.publish(s_raw_msg)
        chunk_lock = False
        conn.sendall(str(int(np.floor(azimuth_global))).encode('utf-8'))
    conn.close()

def main():

    # ros init
    rospy.init_node('sound_localize',anonymous=False)

    import time
    logging.basicConfig(level=logging.DEBUG)
    # create thread for  microphone array beamforming to do localization with TODA or MUSIC
    q = threading.Event()
    t = threading.Thread(target=task, args=(q, ))
    rec_t = threading.Thread(target=rec_thread, args=(q, ))

    rec_t.start()
    t.start()

    # while True:
    #     try:
    #         time.sleep(1.0)
    #     except KeyboardInterrupt:
    #         print('Quit')
    #         q.set()
    #         break
    # wait for the thread
    rospy.spin()
    rec_t.join()
    t.join()

if __name__ == '__main__':

    # ros init
    rospy.init_node('sound_localize',anonymous=False)
    # set socket server client ip and port
    # HOST = '192.168.50.7'  #HOST = '192.168.50.102'    # phone '192.168.43.253' # tealab '192.168.1.176'
    HOST = rospy.get_param("~IP", "192.168.50.7")
    PORT = 8001
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    print ('Server start at: %s:%s' %(HOST, PORT))
    print ('wait for connection...')
    s.listen(5)
    main()
