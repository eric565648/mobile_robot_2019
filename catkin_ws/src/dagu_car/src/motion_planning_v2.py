#!/usr/bin/env python

import rospy
from std_msgs.msg import Int64, Float64MultiArray, Float64
from geometry_msgs.msg import Pose, Quaternion, Point, PoseStamped
import time
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math

from math import fabs, cos, sin

class Controller(object):
    def __init__(self):
    
    
        self.v = 0.0
        self.omega = 0.0
        
        self.cmd_pub = rospy.Publisher("/motors_cmd", Float64MultiArray, queue_size=1)
        self.pos_sub = rospy.Subscriber("/target_pose", PoseStamped, self.planning, queue_size=1)
        
        self.stage = 0
        self.got_response = False
        self.sleep_time = 0.2
        
        self.turn_omega = 1.2
        self.forward_speed = 60
        self.threshold_distance = 2
        self.searching_omega = 1.0
        
        self.facing_angle_limit = 10 * math.pi/180
        
        self.block = True
        self.stop = False
        self.calculating = False
	
        self.interval = 5
        
        self.eps = 1e-10
        
    def planning(self, msg):
        #if self.calculating:
        #    return
    
    
        pos = msg.pose.position
        orientation_q = msg.pose.orientation
        
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion (orientation_list)


        yaw = math.atan2(pos.y, pos.x)
        rospy.loginfo("yaw: %f", yaw)
        
        self.target_angle = yaw
        self.target_pos = [pos.x, pos.y]
        self.got_response = True
        
    def target_decision(self):
        x = self.target_pos[0]
        y = self.target_pos[1]
        yaw = self.target_angle
        while yaw > 2*math.pi:
            yaw -= 2*math.pi
        while yaw < 0:
            yaw += 2*math.pi
            
        yaw_c = yaw + math.pi
        while yaw_c > 2*math.pi:
            yaw_c -= 2*math.pi
        while yaw_c < 0:
            yaw_c += 2*math.pi
        
        angle_range = 30 * math.pi/180
        angle = math.atan2(-y, -x)
        while angle > 2*math.pi:
            angle -= 2*math.pi
        while angle < 0:
            angle += 2*math.pi
        
        diff = (yaw - angle)
        while diff > math.pi:
            diff -= math.pi
        while diff < -math.pi:
            diff += math.pi
        
        if angle_range >= diff >= -angle_range:
            return
        elif math.pi/2 >= diff >= -math.pi/2:
            x = x + self.interval * cos(yaw)
            y = y + self.interval * sin(yaw)
            
            self.target_pos = [x, y]
        elif yaw_c >= angle:
            # is Left
            x = x + self.interval * cos(yaw + math.pi/2)
            y = y + self.interval * sin(yaw + math.pi/2)
            
            self.target_pos = [x, y]
            
        elif yaw_c <= angle:
            # is Right
            x = x + self.interval * cos(yaw - math.pi/2)
            y = y + self.interval * sin(yaw - math.pi/2)
            
            self.target_pos = [x, y]
        else:
            rospy.loginfo("!!! Something got wrong !!!")
        
    def run(self):
        if not self.got_response:
            return
            
        self.calculating = True
        
        if self.block:
            if self.stop:
                self.stop = False
                
                self.v = 0
                self.omega = 0
                
                self.pubMsg()
                time.sleep(self.sleep_time)
                return
                
        #if self.stage == 0:
        #    self.target_decision()
        
        x = self.target_pos[0]
        y = self.target_pos[1]
        
        facing_angle = math.atan2(y, x)
        print("x, y: ", x, y)
        print("face angle: ", facing_angle)


        if x==0:
            self.stage =2
        elif self.facing_angle_limit >= facing_angle >= -self.facing_angle_limit:
            self.stage = 1
        else:
            self.stage = 0
        
        if self.stage == 0:
            self.v = 0
            
            self.omega = y/(fabs(y) + self.eps) * self.turn_omega
            # self.omega = self.turn_omega
            self.sleep_time = 0.1
        elif self.stage == 1:
            if y < self.threshold_distance:
                self.v = 1 * self.forward_speed
            else:
                self.v = self.forward_speed
            self.omega = 0
            
            self.sleep_time = 0.1
        elif self.stage == 2:
            self.v = 0
            self.omega = 0
            self.sleep_time = 0.1
            
        
        self.stop = True
        
        self.pubMsg()
        self.calculating = False

        #time.sleep(self.sleep_time)

    def pubMsg(self):
        #rospy.loginfo('v: %f', self.v)
        #rospy.loginfo('omega: %f', self.omega)

        msg = Float64MultiArray()

        msg.data = [0.0, 0.0]
        msg.data[0] = self.v
        msg.data[1] = self.omega

        self.cmd_pub.publish(msg)

        time.sleep(self.sleep_time)
        msg.data[0] = 0
        msg.data[1] = 0
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
