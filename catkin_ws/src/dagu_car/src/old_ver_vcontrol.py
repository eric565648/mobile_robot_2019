#!/usr/bin/env python

import rospy
from std_msgs.msg import Int64, Float64MultiArray
import RPi.GPIO as GPIO
import time

class Controller(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        rospy.loginfo("[%s] Initializing " %(self.node_name))

	GPIO.setmode(GPIO.BCM)        

        self.light_pin = 18
        GPIO.setup(self.light_pin, GPIO.IN)

        self.left_touch_pin = 17
        self.right_touch_pin = 27
	self.mid_touch_pin = 22

        GPIO.setup(self.left_touch_pin, GPIO.IN)
        GPIO.setup(self.right_touch_pin, GPIO.IN)
	GPIO.setup(self.mid_touch_pin, GPIO.IN)

        self.v = 0.0
        self.omega = 0.0

        self.v_interval = 45
        self.omega_interval = 1.5
        self.time_sleep = 0
        self.idle_time = 15
        self.idle_dir = 1
	self.run_time = 0	

	self.idle_turn = 1

	self.left_hit = 0
	self.right_hit = 0

	self.cmd_pub = rospy.Publisher("/motors_cmd", Float64MultiArray, queue_size=1)
#        self.velocity_pub = rospy.Publisher("motors_v", Float64MultiArray, queue_size=1)
#        self.omega_pub = rospy.Publisher("motors_omega", Int63, queue_size=1)

    def pubMsg(self):
#        rospy.loginfo('v: %f', self.v)
#        rospy.loginfo('omega: %f', self.omega)

	msg = Float64MultiArray()

	msg.data = [0.0, 0.0]
	msg.data[0] = self.v
	msg.data[1] = self.omega

	self.cmd_pub.publish(msg)

    def run(self):
        light_rev = GPIO.input(self.light_pin)

#	print(light_rev)

        right_touch = GPIO.input(self.right_touch_pin)
        left_touch = GPIO.input(self.left_touch_pin)
	mid_touch = GPIO.input(self.mid_touch_pin)

	self.run_time += 1

	if self.run_time >= 300 and light_rev == 0:
	    rospy.loginfo("Priority Light Received")
	    self.v = 90.0
	    self.omega = 0.0

        elif self.time_sleep != 0:
            self.time_sleep -= 1


	elif self.left_hit == 1:
	    self.left_hit = 0
	    self.v = 90.0
	    self.omega = -0.35 * self.omega_interval
	    self.time_sleep = 4


	elif self.right_hit == 1:
	    self.right_hit = 0
	    self.v = 90.0
	    self.omega = 0.35 * self.omega_interval
	    self.time_sleep = 4

	elif mid_touch == 1:
	    rospy.loginfo("Mid touch")
	    self.v = -75.0  

	    self.time_sleep = 2

        elif right_touch == 1:
	    rospy.loginfo("Right touch")	

            self.v = -75.0
            self.omega = 0.0 #1 * self.omega_interval

	    self.right_hit = 1
	
            self.time_sleep = 4

        elif left_touch == 1:
	    rospy.loginfo("Left touch")

            self.v = -75.0
            self.omega = 0.0# * self.omega_interval

	    self.left_hit = 1
            self.time_sleep = 4

        elif light_rev == 0:
	    rospy.loginfo("Light received")
            self.v = 100.0
            self.omega = -0.25 * self.omega_interval

	    self.time_sleep = 6

        else:
	    rospy.loginfo("Idle")
            self.v = 0.0
            self.omega = self.idle_turn * self.omega_interval

            self.idle_time += 1

        if self.idle_time >= 10:
            self.idle_time = 0
            
	    self.idle_turn *= -1
            self.v = 110.0
            self.omega = 0.0

            self.time_sleep = 6

        self.pubMsg()

        time.sleep(0.25)
#        self.run()

if __name__ == '__main__':
    rospy.init_node('vehicle_controller', anonymous=False)
    node = Controller()
#    rospy.on_shutdown(node.on_shutdown)

    while True:
        node.run()
    
    rospy.spin()
