#!/usr/bin/env python

import math
import numpy as np
import rospy
import time
from apriltags2_ros.msg import AprilTagDetectionArray, AprilTagDetection
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Int64
from tf import transformations as tr
import tf

TAG_ID = 40
STATE_MAX = 6

FIND_CREAM = False
FIND_FACE = True
CREAM_ORIGIN_X = 0.3
CREAM_ORIGIN_Y = 0.6

class TargetMux(object):
    """docstring for TargetMux."""
    def __init__(self):
        super(TargetMux, self).__init__()

        rospy.init_node('target_mux')

        # variables
        self.state = 0
        self.face_pose = None
        self.cream_pose = None
        self.robot_pose = None

        self.fg = Pose()

        # Publisher
        self.pubtarget = rospy.Publisher("/target_pose", PoseStamped, queue_size = 1)
        # Subscriber
        subpose = rospy.Subscriber("/orb_slam2_rgbd/pose", PoseStamped, self.robot_cb, queue_size = 1)
        subface = rospy.Subscriber("face_pose", PoseStamped, self.face_cb, queue_size = 1)
        subtag = rospy.Subscriber("tag_detections", AprilTagDetectionArray, self.tag_cb, queue_size = 1)
        substate = rospy.Subscriber("state_input", Int64, self.state_cb, queue_size = 1)

    def robot_cb(self, msg):

        self.robot_pose = msg.pose

        if not FIND_CREAM:
            #print("not find cream")
            if self.cream_pose == None:
                self.cream_pose = Pose()
            trans = [self.robot_pose.position.x,self.robot_pose.position.y,self.robot_pose.position.z]
            rot = [self.robot_pose.orientation.x,self.robot_pose.orientation.y,self.robot_pose.orientation.z,self.robot_pose.orientation.w]
            transformation_matrix = tr.compose_matrix(angles = tr.euler_from_quaternion(rot), translate = trans)
            trans_c = [CREAM_ORIGIN_X,CREAM_ORIGIN_Y,0.08]
            rot_c = [0, 0, 0, 1]
            mat_c = tr.compose_matrix(angles = tr.euler_from_quaternion(rot_c), translate = trans_c)
            inv_mat = tr.inverse_matrix(transformation_matrix)
            new_mat = np.dot(inv_mat, mat_c)
            trans_new = tf.transformations.translation_from_matrix(new_mat)
            self.cream_pose.position.x = trans_new[0]
            self.cream_pose.position.y = trans_new[1]

            #print("cream x: ", self.cream_pose.position.x)
            #print("cream y: ", self.cream_pose.position.y)

    def face_cb(self, msg):

        global FIND_FACE

        if self.state == 3:
            FIND_FACE = True

        self.face_pose = msg.pose

    def tag_cb(self, msg):

        global FIND_CREAM

        for tag in msg.detections:
            if tag.id[0] != TAG_ID:
                continue
            print("get cream")
            FIND_CREAM = True
            self.cream_pose.position.x = tag.pose.pose.pose.position.z
            self.cream_pose.position.y = -1*tag.pose.pose.pose.position.x

    def state_cb(self, msg):

        if msg.data < STATE_MAX:
            self.state = msg.data
            print "Input New State: ", self.state

    def process(self):

        global FIND_FACE

        if self.robot_pose is None:
            return

        print("Now State: ", self.state)

        msg_pose = PoseStamped()

        # find human face
        if self.state == 0:
            if self.face_pose == None:
                return
            msg_pose.pose = self.face_pose

            if -0.23 < self.face_pose.position.x < 0.23 and -0.2 < self.face_pose.position.y < 0.2:
                self.state = 1
                self.fg.position.x = self.robot_pose.position.x
                self.fg.position.y = self.robot_pose.position.y
                FIND_FACE = False

        elif self.state == 1:
            msg_pose.pose.position.x = 0
            self.pubtarget.publish(msg_pose)
            print("state 1 sleep")
            time.sleep(7)
            self.state = 2

        # find cream
        elif self.state == 2:
            if self.cream_pose == None:
                return
            msg_pose.pose = self.cream_pose

            print("cx cy", self.cream_pose.position.x, self.cream_pose.position.y)

            if -0.23 < self.cream_pose.position.x < 0.23 and -0.05 < self.cream_pose.position.y < 0.05:
                self.state = 3

        elif self.state == 3:

            print("State 3")
            msg_pose.pose.position.x = 0.1
            msg_pose.pose.position.y = 0
            self.pubtarget.publish(msg_pose)
            time.sleep(3)
            msg_pose.pose.position.x = 0
            self.pubtarget.publish(msg_pose)
            print("state 3 sleep")
            time.sleep(7)
            self.state = 4

        elif self.state == 4:
            if self.face_pose == None:
                return

            if FIND_FACE == True:
                msg_pose.pose = self.face_pose
            else:
                trans = [self.robot_pose.position.x,self.robot_pose.position.y,self.robot_pose.position.z]
                rot = [self.robot_pose.orientation.x,self.robot_pose.orientation.y,self.robot_pose.orientation.z,self.robot_pose.orientation.w]
                transformation_matrix = tr.compose_matrix(angles = tr.euler_from_quaternion(rot), translate = trans)
                trans_c = [self.fg.position.x,self.fg.position.y,0.08]
                rot_c = [0, 0, 0, 1]
                mat_c = tr.compose_matrix(angles = tr.euler_from_quaternion(rot_c), translate = trans_c)
                inv_mat = tr.inverse_matrix(transformation_matrix)
                new_mat = np.dot(inv_mat, mat_c)
                trans_new = tf.transformations.translation_from_matrix(new_mat)
                msg_pose.pose.position.x = trans_new[0]
                msg_pose.pose.position.y = trans_new[1]

            print("face x: ", trans_new[0])
            print("face y: ", trans_new[1])
            if -0.23 < msg_pose.pose.position.x < 0.23 and -0.23 < msg_pose.pose.position.y < 0.2:
                self.state = 5

        elif self.state == 5:
            msg_pose.pose.position.x = 0

        self.pubtarget.publish(msg_pose)


if __name__ == '__main__':
    d = TargetMux()

    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        d.process()
        r.sleep()
