#!/usr/bin/env python

import rospy
from std_msgs.msg import Int64, Float64MultiArray, Float64
import RPi.GPIO as GPIO
import time

class Controller(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        rospy.loginfo("[%s] Initializing " %(self.node_name))

	GPIO.setmode(GPIO.BCM)        

        self.light_pin = 18
        GPIO.setup(self.light_pin, GPIO.IN)

        self.left_touch_pin = 22
        self.right_touch_pin = 27
	self.mid_touch_pin = 17

        GPIO.setup(self.left_touch_pin, GPIO.IN)
        GPIO.setup(self.right_touch_pin, GPIO.IN)
	GPIO.setup(self.mid_touch_pin, GPIO.IN)

	self.cmd_pub = rospy.Publisher("/motors_cmd", Float64MultiArray, queue_size=1)
	self.freq_sub = rospy.Subscriber("/becon_freq", Float64, self.becon_listener)

	self.v = 0.0
        self.omega = 0.0

        self.idle_speed = 1.2
        self.idle_dir = -1

        self.turn_omega = -1.2
        self.backward_speed = -95
        self.forward_speed = 80

        self.stage = 0

	self.searching_omega = 1.32
	self.searching_time = 0

        self.idle_time = 0
        self.sleep_time = 0.75
        self.finding_time = 0	
  
	self.becon_found = 1

	# 0 -> 1500; 1 -> 600
	self.gate_number = 1

	self.stop_sign = 1

	self.stage_for_retreat = 0
	self.idle_time_for_retreat = 0

	self.searching_dir = 1
        self.becon_found_once = False
        self.spin_time = 0
        self.searching_becon = False

	self.init_lamda = 1.5
        self.searching_lamda = self.init_lamda

	self.first = True
	self.becon_stay_time = 0
	self.searching_limit = 15

	self.stuck_time = 0
    def becon_listener(self, msg):
	freq = msg.data

	if self.gate_number == 1:
	   if  freq >= 0.27 and freq <= 0.32:
		rospy.loginfo("becon found")
		self.becon_found = 0
	   else:
		self.becon_found = 1
	else:
	    if freq >= 0.17 and freq <= 0.22:
		rospy.loginfo("becon found")
		self.becon_found = 0
	    else:
		self.becon_found = 1

    def retreat(self, r, l):

	self.becon_stay_time += 1
	if self.becon_stay_time >= 1000:
	    self.becon_found_once = False
	
        if self.becon_found_once:
	    if self.stop_sign == 0:
		self.stop_sign = 1
	    else:
		self.stop_sign = 0

            if r == 1 or l == 1:
		rospy.loginfo("Hit go back and search again")

                self.searching_lamda = self.init_lamda
                self.v = self.backward_speed*0.8
                self.omega = 0.8
                self.searching_becon = True

                self.sleep_time = 0.6

            elif self.becon_found == 0:
		rospy.loginfo("Becon Found Forward")

                self.searching_lamda = self.init_lamda
                self.v = self.forward_speed
                self.omega = 0

                self.sleep_time = 0.05

            elif self.searching_becon:
		rospy.loginfo("Finding Becon")

                if self.searching_time < self.searching_limit:
                    self.v = 0
                    self.omega = 0.8 * self.idle_speed * self.searching_lamda

		    self.searching_time += 1
                    self.sleep_time = 0.05
                elif self.searching_time < self.searching_limit*2:
                    self.v = 0
                    self.omega = -0.8 * self.idle_speed * self.searching_lamda

		    self.searching_time += 1
                    self.sleep_time = 0.05
                else:
                    self.searching_time = 0

		    self.searching_limit += 15
                    self.searching_lamda *= 1.01
                    self.sleep_time = 0.001
            else:
                self.searching_lamda = self.init_lamda
                self.searching_becon = True

                self.sleep_time = 0.001
        else:
	    rospy.loginfo("Still Finding Becon")

            self.searching_time = 0
            if r == 1 or l == 1:
                self.v = self.backward_speed
                self.omega = 0

                self.sleep_time = 0.4
                self.spin_time = 0
            elif self.becon_found == 0:
                self.becon_found_once = True

                self.sleep_time = 0.002
            else:
                if self.spin_time <= 100:
		    rospy.loginfo("Spinning")
                    self.v = 0
                    self.omega = self.idle_speed * 1.35

                    self.spin_time += 1
                    self.sleep_time = 0.05
                else:
                    self.v = self.forward_speed
                    self.omega = 0

                    self.spin_time = 0
                    self.sleep_time = 1.25

        self.pubMsg()
        time.sleep(self.sleep_time)

    def run(self):
        light_rev = GPIO.input(self.light_pin)

#	print(light_rev)
	self.stuck_time = 0

        right_touch = GPIO.input(self.right_touch_pin)
        left_touch = GPIO.input(self.left_touch_pin)
	got_target = GPIO.input(self.mid_touch_pin)

#	light_rev = 1
#	got_target = 1
	if got_target == 1:
	    rospy.loginfo("Got the target Finding the way out")
            self.retreat(right_touch, left_touch)
	    return 
            #self.stage = 0
	    #rospy.loginfo("Target Got Finding Gate")
            light_rev = self.becon_found
	    self.backward_speed *= 1
	else:
	    self.backward_speed = -95

        if self.stage != 0:
	    self.stuck_time = 0
	    self.idle_time = 0

            if self.stage == 1:
		rospy.loginfo("Stage 1")
                # turn 180 degree

                self.v = 0.0
                self.omega = 1.2 * self.idle_speed
                self.sleep_time = 0.8

                self.stage = 0
            elif self.stage == 2:
		rospy.loginfo("Stage 2")
                # turn left
                self.v = self.forward_speed
                self.omega = -0.85 * self.turn_omega
                self.sleep_time = 0.8

                self.stage = 0
            elif self.stage == 3:
		rospy.loginfo("Stage 3")
                # turn right
                self.v = self.forward_speed
                self.omega = 0.85 * self.turn_omega
                self.sleep_time = 0.8

                self.stage = 0
            elif self.stage == 4:
		rospy.loginfo("Stage 4")
                # stop and find the direction
                # but first detect obstacle
                if right_touch == 1:
                    self.stage = 2
                elif left_touch == 1:
                    self.stage = 3

                elif light_rev == 0:
                    # can still find the light
                    # means the direction is right
                    self.stage = 5
                else:
                    # does not find light
                    # reverse the rotate direction
                    if self.finding_time >= 10:
                        self.stage = 0
                        self.finding_time = 0
                    else:
                        self.stage = 6
                        self.finding_time += 1

                self.sleep_time = 0.1
            elif self.stage == 5:
		rospy.loginfo("Stage 5")
                self.v = self.forward_speed
		self.omega = 0

		self.sleep_time = 0.01
                self.stage = 7
            elif self.stage == 6:
		rospy.loginfo("Stage 6")

                self.v = 0
                self.omega = -1 * self.idle_dir * self.idle_speed * 0.75
                self.sleep_time = 0.125

                self.stage = 4
	    elif self.stage == 7:
		rospy.loginfo("Stage 7")
		if right_touch or left_touch:
		    self.stage = 8
		    self.v = self.backward_speed
		    self.omega = 0
		    self.sleep_time = 1.0

		elif light_rev == 0:
		    self.stage = 5
		    self.sleep_time = 0.002
		else:
		    if self.searching_time >= 80:
			self.stage = 0
			self.searching_time = 0
		    else:
			self.stage = 8
		    self.sleep_time = 0.002

	    elif self.stage == 8:
		rospy.loginfo("Stage 8")

		self.v = 0
		if self.searching_time < 60:
		    self.omega = self.searching_omega
		elif self.searching_time < 120:
		    self.omega = -1 * self.searching_omega
		
		self.stage = 7	
		self.searching_time += 1

		self.sleep_time = 0.025

        # both touch
        elif right_touch and left_touch:
	    rospy.loginfo("Two Touch")
            self.stage = 1

            # stage 1
            # go backward and turn 180 degree
            self.v = -95.0
            self.omega = 0.0
            self.sleep_time = 0.75

        # right touch
        elif right_touch == 1:
	    rospy.loginfo("Right Touch")	
            self.stage = 2

            # stage 2
            # go backward and turn left
            self.v = self.backward_speed
            self.omega = 0.0
            self.sleep_time = 0.75

	    self.stuck_time = 0
        # left touch
        elif left_touch == 1:
	    rospy.loginfo("Left Touch")
            self.stage = 3

            # stage 3 =
            # go backward and turn right
            self.v = self.backward_speed
	    self.omega = 0
            self.sleep_time = 0.75

	    self.stuck_time = 0

        # light found
        elif light_rev == 0:
	    rospy.loginfo("Light Found")

            self.stage = 4
            # stage 4
            # find light but need to confirm
            self.v = 0
            self.omega = 0
            self.sleep_time = 0.001

	    self.stuck_time = 0
        # idle
        else:
	    rospy.loginfo("Idle")
            self.stage = 0
            # stage 0
            # idle and rotate

            if self.idle_time <= 20:
                self.v = self.forward_speed
                self.omega = 0

                self.idle_time += 1

		if self.first:
		    self.sleep_time = 0.001
		    self.first = False
		else:
		    self.sleep_time = 0.15
            elif self.idle_time <= 100:
                self.v = 0
                self.omega = self.idle_dir * self.idle_speed

                self.idle_time += 1
		self.sleep_time = 0.05
            else:
                self.idle_dir *= -1
                self.idle_time = 0
		self.sleep_time = 0.002

	    self.stuck_time += 1

	    if self.stuck_time >= 200:
		rospy.loginfo("Stuck Debugging")
		self.v = self.backward_speed
		self.omega = 0
		self.stage = 1

		self.sleep_time = 2.0
		self.stuck_time = 0

            #self.sleep_time = 0.15

        self.pubMsg()

        time.sleep(self.sleep_time)

    def pubMsg(self):
#        rospy.loginfo('v: %f', self.v)
#        rospy.loginfo('omega: %f', self.omega)

        msg = Float64MultiArray()

        msg.data = [0.0, 0.0]
        msg.data[0] = self.v
        msg.data[1] = self.omega

        self.cmd_pub.publish(msg)
    def on_shutdown(self):
	self.v = 0
	self.omega = 0
	self.pubMsg()

if __name__ == '__main__':
    rospy.init_node('vehicle_controller', anonymous=False)
    node = Controller()
    rospy.on_shutdown(node.on_shutdown)

    while not rospy.is_shutdown():
        node.run()
    
    rospy.spin()
