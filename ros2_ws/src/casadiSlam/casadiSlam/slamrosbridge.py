import casadi
import numpy as np
from rclpy.node import Node
import rclpy
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage

class slamNode(Node):

    def __init__(self):
        super().__init__('slamNode')
        self.subscriber_ = self.create_subscription(PoseStamped, 'cam_drone', 10)
        self.get_logger().info('connected')
        


def main(args=None):
    rclpy.init(args=args)

    slam_node =slamNode()

    rclpy.spin(slam_node)
    #rclpy.shutdown()


if __name__ == '__main__':
    main()