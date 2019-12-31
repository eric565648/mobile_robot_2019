#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool, Int8
from std_srvs.srv import Empty

class fsm_demo(object):
    """docstring for fsm_demo."""
    def __init__(self):
        super(fsm_demo, self).__init__()

        rospy.init_node('farm_demo_fsm')

        # The current state
        self.state = 0

        # Service Clients
        rospy.wait_for_service('pick_process')
        self.pick = rospy.ServiceProxy('pick_process', Empty)
        rospy.wait_for_service('to_dispose')
        self.to_dispose = rospy.ServiceProxy('to_dispose', Empty)
        rospy.wait_for_service('place_process')
        self.place = rospy.ServiceProxy('place_process', Empty)

        # Publisher
        self.pubber = rospy.Publisher("/farm_demo_states", Int8, queue_size=5)
        self.pub_states()
        # Subscriber
        self.sub_user_input = rospy.Subscriber("/fsm_user_inputs", Int8, self.user_input_cb, queue_size=5)

    def main_loop(self):

        if self.state == 1:
            while not rospy.is_shutdown:
                if self.state == 1:
                    try:
                        self.pick()
                        self.state == 2
                    except Exception as e:
                        rospy.loginfo("Pick Failed")
                elif self.state == 2:
                    try:
                        self.to_dispose()
                        self.state == 3
                    except Exception as e:
                        rospy.loginfo("To Depose Failed")
                elif self.state == 3:
                    try:
                        self.place()
                        self.state == 0
                    except Exception as e:
                        rospy.loginfo("Place Failed")

    def user_input_cb(self, msg):

        state = msg.data

        if (state != 1) || (state != 4):
            rospy.loginfo("Initial state should be 1 or 4")
            return
        elif self.state != 0:
            rospy.loginfo("User have no priority now.")
            return
        else:
            self.state = state
            self.pub_states()
            self.main_loop()

    def pub_states():
        msg = Int8()
        msg.data = self.state
        rospy.loginfo("To State: ", str(self.state))

if __name__ == '__main__':
    d = fsm_demo()
    rospy.spin()
