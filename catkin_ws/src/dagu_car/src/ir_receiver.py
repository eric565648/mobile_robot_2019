#!/usr/bin/env python

import rospy
from std_msgs.msg import Int64, Float64
import RPi.GPIO as GPIO
import time
import threading


class IR_Reciever(object):
    def __init__(self):
        self.input_pin = 25

#        self.t = threading.Thread(name='counter', target=self.service)

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.input_pin, GPIO.IN)

        self.freq_pub = rospy.Publisher("/becon_freq", Float64, queue_size=1)
        self.zeros = 0
        self.total = 1
        self.freq = 0.0

	self.time = 0

#	self.t.start()

    def run(self):
        if not GPIO.input(self.input_pin):
            self.zeros += 1
        self.total += 1
	
	if self.total >= 120:
	    self.service()

	time.sleep(0.001)

    def service(self):
        self.freq = 1.0 * self.zeros / self.total
        self.zeros = 0
        self.total = 1
#        rospy.loginfo("Publishing Freq: %lf", self.freq)

        msg = Float64()
        msg.data = self.freq
        self.freq_pub.publish(msg)

        time.sleep(0.002)
#        self.service()


if __name__ == '__main__':
    rospy.init_node('IR_Receiver', anonymous=False)
    node = IR_Reciever()
#    rospy.on_shutdown(node.on_shutdown)

    while not rospy.is_shutdown():
        node.run()

    rospy.spin()
