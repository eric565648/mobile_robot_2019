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
from std_msgs.msg import Int16MultiArray
from sound_localize.msg import SoundRaw

logger              = logging.getLogger('MicArray')

chunk_buffer = None
chunk_lock = True
azimuth_global = 0
chunk_time = None

def rec_thread(quit_event):
    conn, addr = s.accept()

    pubFrameRaw = rospy.Publisher('sound_raw', SoundRaw, queue_size=1)

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
        print("size: ", xx.shape)
        # print("The xx: ", xx)
        # print("xx len: ", xx.size)
        s_raw_msg = SoundRaw()
        s_raw_msg.data = xx
        s_raw_msg.header.stamp = chunk_time
        pubFrameRaw.publish(s_raw_msg)
    conn.close()

def main():

    # ros init
    rospy.init_node('sound_localize',anonymous=False)

    import time
    logging.basicConfig(level=logging.DEBUG)
    # create thread for  microphone array beamforming to do localization with TODA or MUSIC
    q = threading.Event()
    rec_t = threading.Thread(target=rec_thread, args=(q, ))

    rec_t.start()

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
