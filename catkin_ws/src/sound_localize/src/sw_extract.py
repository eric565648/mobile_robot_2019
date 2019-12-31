#!/usr/bin/env python

import rospy
import numpy as np
import math
from sound_localize.msg import SoundBearing
from pozyx_ros.msg import DeviceRangeArray
from anchor_measure.msg import PoseDirectional # pose with direction
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import Marker
import tf
import message_filters

class SWBallExtract(object):
    """docstring for SWBallExtract."""
    def __init__(self):
        super(SWBallExtract, self).__init__()

        # variables
        self.robot_tags = rospy.get_param("~robot_tags", [0, 1])
        self.visualization = rospy.get_param("~visualization", True)

        # Publisher
        self.sw_pub = rospy.Publisher("sw_poses", PoseDirectional, queue_size = 3)
        # tf
        self.br = tf.TransformBroadcaster()

        ## msg filter sync range and bearing
        range_sub1 = message_filters.Subscriber('pozyx_range_array', DeviceRangeArray)
        bearing_sub1 = message_filters.Subscriber('filtered_bearing', SoundBearing)
        # store a lot of range, cuz range has high freq and bearing need time to process
        ts1 = message_filters.ApproximateTimeSynchronizer([range_sub1, bearing_sub1], 120, slop=0.1)
        ts1.registerCallback(self.sw_cb)

        # visualization parameters
        self.marker_pub = rospy.Publisher('marker_sw', Marker, queue_size=3)
        self.marker = Marker(type=Marker.SPHERE_LIST, ns='sw_anchor', action=Marker.ADD)
        self.marker.header.frame_id = 'map'
        self.marker.scale.x = 0.1
        self.marker.scale.y = 0.1
        self.marker.color.a = 1.0
        self.marker.color.g = 1

        self.points_list = []

    def sw_cb(self, range_msg, bearing_msg):

        for device in range_msg.rangeArray:

            if device.tag_id in self.robot_tags: # if they are tags on any robots
                continue

            x, y, z = self.getCartesian(device.distance/1000., -1*bearing_msg.azimuth, bearing_msg.elevation+90)

            anchor_msg = PoseDirectional()
            anchor_msg.to.data = str(device.tag_id)
            # anchor_msg.from.data = '' # weird invalid syntax problem
            anchor_msg.pose.header.stamp = bearing_msg.header.stamp
            anchor_msg.pose.pose.position.x = x
            anchor_msg.pose.pose.position.y = y
            anchor_msg.pose.pose.position.z = z

            print("Point: ")
            print("x: ",x)
            print("y: ",y)
            print("z: ",z)
            print("---------------------")

            self.sw_pub.publish(anchor_msg)

            # self.br.sendTransform((anchor_msg.pose.pose.position.x, \
            #                      anchor_msg.pose.pose.position.y, anchor_msg.pose.pose.position.z), \
            #                     (target_pose.orientation.x, target_pose.orientation.y, \
            #                     target_pose.orientation.z, target_pose.orientation.w), \
            #                     anchor_msg.pose.header.stamp, str(device.tag_id), 'base_link')

            self.points_list = np.append(self.points_list, anchor_msg.pose.pose.position)
            if self.visualization:
                self.visualize_points(anchor_msg.pose.header.stamp)

    def getCartesian(self, r, theta, phi):

        # from spherical to cartisian
        # r : distance
        # theta : azimuth angle
        # phi : elevation angle

        r_xy = r * math.sin(math.radians(phi))
        x = r_xy * math.cos(math.radians(theta))
        y = r_xy * math.sin(math.radians(theta))
        z = r * math.cos(math.radians(phi))

        return x, y, z;

    def visualize_points(self, timestamp):

        self.marker.points = self.points_list
        self.marker.header.stamp = timestamp
        self.marker.lifetime = rospy.Duration()
        self.marker_pub.publish(self.marker)

if __name__ == '__main__':
	rospy.init_node('sw_extract',anonymous=False)
	swe = SWBallExtract()
	rospy.spin()
