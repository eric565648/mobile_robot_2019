#!/usr/bin/env python

import rospy
from std_msgs.msg import Int64, Float64MultiArray, Float64
from geometry_msgs.msg import Pose, Quaternion, Point
import RPi.GPIO as GPIO
import time
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math

class Controller(object):
    def __init__(self):
    
    
        self.v = 0.0
        self.omega = 0.0
        
        self.cmd_pub = rospy.Publisher("/motors_cmd", Float64MultiArray, queue_size=1)
        self.pos_sub = rospy.Subscriber("", Pose, self.planning)
        
        self.stage = 0
        self.got_response = False
        self.sleep_time = 0.2
        
        self.turn_omega = 1.2
        self.forward_speed = 90
        self.threshold_distance = 2
        self.searching_omega = 1.0
        
        self.facing_angle_limit = 15 * math.pi/180
        
        self.block = True
        self.stop = False
        
        self.eps = 1e-10
        
    def planning(self, msg):
        self.got_response = True
    
        pos = msg.position
        orientation_q = msg.orientation
        
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion (orientation_list)
        ros.loginfo("yaw: %f", yaw)
        
        self.target_angle = yaw
        self.target_pos = [pos.x, pos.y]
        
    def run(self):
        if not self.got_response:
            return
        
        if self.block:
            if self.stop:
                self.stop = False
                
                self.v = 0
                self.omega = 0
                
                self.pubMsg()
                time.sleep(self.sleep_time)
                return
                
        
        x = self.target_pos[0]
        y = self.target_pos[1]
        
        facing_angle = math.atan2(y, x)
        
        if self.facing_angle_limit >= facing_angle - (math.pi/2) >= -self.facing_angle_limit:
            self.stage = 1
        elif self.stage == 1:
            self.stage = 2
        else:
            self.stage = 0
        
        if self.stage == 0:
            self.v = 0
            
            self.omega = -x/(fabs(x) + self.eps) * self.turn_omega
            self.sleep_time = 0.1
            
        elif self.stage == 1:
            if y < self.threshold_distance:
                self.v = 0.75 * self.forward_speed
            else:
                self.v = self.forward_speed
            self.omega = 0
            
            self.sleep_time = 0.1
            
        elif self.stage == 2:
            self.v = 0.8 * self.forward_speed 
            self.omega = -x/(fabs(x) + self.eps) * self.searching_omega
            
            self.sleep_time = 0.1
        
        self.stop = True
        
        self.pubMsg()

        time.sleep(self.sleep_time)

    def pubMsg(self):
        #rospy.loginfo('v: %f', self.v)
        #rospy.loginfo('omega: %f', self.omega)

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
