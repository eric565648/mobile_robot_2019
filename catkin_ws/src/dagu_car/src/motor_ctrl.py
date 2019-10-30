#!/usr/bin/env python

import rospy
from std_msgs.msg import Int64

def listener_L(msg):
    print "pub L"
    pub_L = rospy.Publisher('/pwm_L', Int64, queue_size=1)
    pub_L.publish(msg)

def listener_R(msg):
    print "pub R"
    pub_R = rospy.Publisher('/pwm_R', Int64, queue_size=1) 
    pub_R.publish(msg)

if __name__ == '__main__':
    rospy.init_node('motor_control_node', anonymous=False)

    rospy.Subscriber("/left_cmd", Int64, listener_L)
    rospy.Subscriber("/right_cmd", Int64, listener_R)

    rospy.spin()
