#! /usr/bin/env python
import rospy
import tf
from geometry_msgs.msg import Pose, TransformStamped, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header


class PoseOdomBridge():
	def __init__(self):
		self.subpose = rospy.Subscriber("/orb_slam2_rgbd/pose", PoseStamped, self.cb, queue_size = 10)
		self.pubodom = rospy.Publisher("/vo_odom", Odometry, queue_size = 10)

	def cb(self, msg):

		odom = Odometry()
		odom.header = Header()
		odom.header = msg.header
 
		odom.pose.pose = msg.pose
		self.pubodom.publish(odom)
	def on_shutdown(self):
		pass

if __name__ == "__main__":
	rospy.init_node('PoseOdomBridge')
	node = PoseOdomBridge()
	rospy.on_shutdown(node.on_shutdown)
	rospy.spin()
