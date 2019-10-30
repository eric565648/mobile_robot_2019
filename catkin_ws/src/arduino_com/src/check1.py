#!/usr/bin/env python

import rospy
from std_msgs.msg import Int64

def waitinput():
    userinput = input("Put a NUUUUMMMMMMBBBBBEEEEERRRRR: ")
    to_send = int(userinput)
    msg = Int64()
    msg.data = to_send
    pub.publish(msg.data)
    

def callback(data):
    rospy.loginfo('Answer: %s', str(data.data))
    waitinput()



if __name__ == '__main__':
    rospy.init_node('check1', anonymous=True)

    pub = rospy.Publisher('to_add', Int64, queue_size=1)
    rospy.Subscriber('result', Int64, callback)
    try:
        waitinput()
    except rospy.ROSInterruptException:
        pass

    rospy.spin()


