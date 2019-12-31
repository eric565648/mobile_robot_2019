#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from geometry_msgs.msg import PoseArray, PoseStamped

class FaceDetection(object):
    """docstring for face_detection."""
    def __init__(self):
        super(face_detection, self).__init__()

        # Subscriber
        ## msg filter sync rgb and d images
        image_sub1 = message_filters.Subscriber('camera_middle_1/color/image_raw', Image)
        depth_sub1 = message_filters.Subscriber('camera_middle_1/aligned_depth_to_color/image_raw', Image)
        ts1 = message_filters.ApproximateTimeSynchronizer([image_sub1, depth_sub1], 10, slop=0.1)
        ts1.registerCallback(self.img_cb)

        # publisher
        self.face_pub = rospy.Publisher('face_pose', PoseStamped, queue_size=1)

    def img_cb(self, rgb_msg, depth_msg):

        rospy.loginfo("Get Image")

        cv_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        cv_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        img_bb = cv_image.copy()

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        gray = cv2.cvtColor(img_bb, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        faces_points = []
        for (x,y,w,h) in faces:
            zc = cv_depth[(y+h/2), (x+w/2)]
            if zc == 0:
                print("Face too far")
                continue

            zc = float(zc)/10000. # SR300 is a bit weird
            rx, ry, rz = self.getXYZ((x+w/2), (y+h/2), zc)
            if rx < faces_points[0]:
                faces_points = np.array([rx, ry, rz])

        fp = PoseStamped()
        fp.header.stamp = rgb_msg.header.stamp
        fp.pose.position.x = faces_points[0]
        fp.pose.position.y = faces_points[1]
        fp.pose.position.z = faces_points[2]
        self.face_pub.publish(fp)

    def getXYZ(self, x, y, zc):

        x = float(x)
        y = float(y)
        zc = float(zc)
        inv_fx = 1.0/self.fx;
    	inv_fy = 1.0/self.fy;
    	x = (x - self.cx) * zc * inv_fx;
    	y = (y - self.cy) * zc * inv_fy;
    	return zc, -1*x, -1*y;


if __name__ == '__main__':
    f = FaceDetection()
    rospy.spin()
