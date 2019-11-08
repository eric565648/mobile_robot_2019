#!/usr/bin/env python

import rospy
from std_msgs.msg import Int64

def listener_L(msg):
#    print "pub L"
    pub_L = rospy.Publisher('/pwm_L', Int64, queue_size=1)
    pub_L.publish(msg)

def listener_R(msg):
#    print "pub R"
    pub_R = rospy.Publisher('/pwm_R', Int64, queue_size=1) 
    pub_R.publish(msg)

def pubMsg(vr, vl):
    rospy.loginfo('right: %d', vr)
    rospy.loginfo('left: %d', vl)

    msg_vr = Int64()
    msg_vr.data = vr
    listener_R(msg_vr)

    msg_vl = Int64()
    msg_vl.data = vl
    listener_L(msg_vl)

if __name__ == '__main__':
    rospy.init_node('motor_control_node', anonymous=False)

    rospy.Subscriber("/left_cmd", Int64, listener_L)
    rospy.Subscriber("/right_cmd", Int64, listener_R)

    v = 0
    omega = 0
    gain = 1
    trim = 0
    l = 0.5
    base_line = 80
    rospy.loginfo('start')

    while True:
	gain = float(raw_input("right: "))
	trim = float(raw_input("left: "))

	vr = 100
	vl = 100
	vr = (gain + trim) * vr
	vl = (gain - trim) * vl

	pubMsg(vr, vl)	

    rospy.spin()


