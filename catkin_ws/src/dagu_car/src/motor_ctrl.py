#!/usr/bin/env python

import rospy
from std_msgs.msg import Int64, Float64MultiArray

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

def listener(msg):

    gain = 2
    trim = 0.1
    l = 0.5
    base_line = 80

    lin_v = msg.data[0]
    omega_v = msg.data[1]

    vr = (gain + trim) * (lin_v + 0.5*omega_v*base_line)
    vl = (gain - trim) * (lin_v - 0.5*omega_v*base_line)

    pubMsg(vr, vl)

if __name__ == '__main__':
    rospy.init_node('motor_control_node', anonymous=False)

    rospy.Subscriber("/motors_cmd", Float64MultiArray, listener)
#    rospy.Subscriber("/motors_omega", Int64, listener_R)

    v = 0
    omega = 0
    gain = 1
    trim = 0
    l = 0.5
    base_line = 80
    rospy.loginfo('start')

    while False:
	gain = float(raw_input("gain: "))
	trim = float(raw_input("trim: "))
	lin_v = float(raw_input("linear velocity: "))
	omega_v = float(raw_input("angular omega: "))
	t_sec = float(raw_input("duration (sec): "))

	#vr = 100
	#vl = 100
	vr = (gain + trim) * (lin_v + 0.5*omega_v)
	vl = (gain - trim) * (lin_v - 0.5*omega_v)

	pubMsg(vr, vl)

	rospy.sleep(t_sec)

	pubMsg(0, 0)

    rospy.spin()


