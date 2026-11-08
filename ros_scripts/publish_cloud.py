#!/usr/bin/env python3

import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from plyfile import PlyData


class CloudPublisher(Node):

    def __init__(self):
        super().__init__("valmet_cloud_publisher")

        path = (
            "outputs/ros/valmet_cloud.ply"
        )

        ply = PlyData.read(path)
        v = ply["vertex"]

        xyz = np.column_stack([
            v["x"], v["y"], v["z"]
        ]).astype(np.float32)

        header = Header()
        header.frame_id = "map"

        self.cloud = point_cloud2.create_cloud_xyz32(
            header,
            xyz.tolist()
        )

        self.publisher = self.create_publisher(
            PointCloud2,
            "/valmet/pointcloud",
            1
        )

        self.timer = self.create_timer(
            2.0,
            self.publish_cloud
        )

        self.get_logger().info(
            f"Loaded {len(xyz)} points"
        )

    def publish_cloud(self):
        self.cloud.header.stamp = (
            self.get_clock().now().to_msg()
        )

        self.publisher.publish(self.cloud)


def main():
    rclpy.init()
    node = CloudPublisher()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
