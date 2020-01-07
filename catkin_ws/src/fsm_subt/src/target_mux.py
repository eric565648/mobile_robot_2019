#!/usr/bin/env python

import rospy
from apriltags2_ros.msg import AprilTagDetectionArray, AprilTagDetection
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Int64

TAG_ID = 123
STATE_MAX = 6

class TargetMux(object):
    """docstring for TargetMux."""
    def __init__(self):
        super(TargetMux, self).__init__()

        rospy.init_node('target_mux')

        # variables
        self.state = 0
        self.face_pose = None
        self.cream_pose = Pose()
        self.cream_pose.x = 0.5
        self.cream_pose.y = 0.8
        self.robot_pose = None


        # Publisher
        self.pubtarget = rospy.Publisher("/target_pose", PoseStamped, queue_size = 1)
        # Subscriber
        subpose = rospy.Subscriber("/orb_slam2_rgbd/pose", PoseStamped, self.robot_cb, queue_size = 10)
        subface = rospy.Subscriber("face_pose", PoseStamped, self.face_cb, queue_size = 10)
        subtag = rospy.Subscriber("tag_detections", AprilTagDetectionArray, self.tag_cb, queue_size = 10)
        substate = rospy.Subscriber("state_input", Int64, self.state_cb, queue_size = 1)

    def robot_cb(self, msg):

        self.robot_pose = msg.pose

    def face_cb(self, msg):

        self.face_pose = msg.pose

    def tag_cb(self, msg):

        for tag in msg.detections:
            if tag.id[0] != TAG_ID:
                continue
            self.cream_pose = tag.pose.pose

    def state_cb(self, msg):

        if msg.data < STATE_MAX:
            self.state = msg.data
            print "New State: ", self.state

    def process(arg):

        if self.robot_pose is None:
            return

        msg_pose = PoseStamped()

        # find human face
        if self.state == 0:
            if self.face_pose == None:
                return
            msg_pose.pose = self.face_pose

            if self.face_pose.position.x < 0.2 and self.face_pose.position.y < 0.2:
                self.state = 1
        # find cream
        elif self.state == 2:
            msg_pose.pose = self.cream_pose

            if self.face_pose.position.x < 0.02 and self.face_pose.position.y < 0.02:
                self.state = 3

        if self.state == 3:
            if self.face_pose == None:
                return
            msg_pose.pose = self.face_pose

            if self.face_pose.position.x < 0.2 and self.face_pose.position.y < 0.2:
                self.state = 1

if __name__ == '__main__':
    d = TargetMux()

    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        d.process()
        r.sleep()
