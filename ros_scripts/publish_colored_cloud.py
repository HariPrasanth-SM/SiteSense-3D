#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from sensor_msgs.msg import PointCloud2, PointField


def load_cloud(path):
    """Load XYZRGB data from a compressed NumPy file."""

    print(f"Loading point cloud: {path}")

    with np.load(path) as data:
        xyz = data["xyz"].astype(np.float32)
        rgb = data["rgb"].astype(np.uint8)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Expected xyz shape (N, 3)")

    if rgb.shape != xyz.shape:
        raise ValueError("Expected rgb shape (N, 3)")

    print(f"Loaded {len(xyz):,} points")

    return xyz, rgb


def create_pointcloud2(xyz, rgb, frame_id="map"):
    """
    Efficiently create a PointCloud2 message.

    Each point:
        x: float32
        y: float32
        z: float32
        rgb: packed uint32

    Total: 16 bytes per point.
    """

    n = len(xyz)

    packed_rgb = (
        (rgb[:, 0].astype(np.uint32) << 16)
        | (rgb[:, 1].astype(np.uint32) << 8)
        | rgb[:, 2].astype(np.uint32)
    )

    dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("rgb", "<u4"),
    ])

    points = np.empty(n, dtype=dtype)

    points["x"] = xyz[:, 0]
    points["y"] = xyz[:, 1]
    points["z"] = xyz[:, 2]
    points["rgb"] = packed_rgb

    msg = PointCloud2()

    msg.header.frame_id = frame_id

    msg.height = 1
    msg.width = n

    msg.fields = [
        PointField(
            name="x",
            offset=0,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="y",
            offset=4,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="z",
            offset=8,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="rgb",
            offset=12,
            datatype=PointField.UINT32,
            count=1,
        ),
    ]

    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * n
    msg.is_dense = bool(np.isfinite(xyz).all())

    msg.data = points.tobytes()

    return msg


class ColoredCloudPublisher(Node):

    def __init__(self, args):
        super().__init__("valmet_colored_cloud_publisher")

        xyz, rgb = load_cloud(args.file)

        self.cloud = create_pointcloud2(
            xyz,
            rgb,
            frame_id=args.frame,
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.publisher = self.create_publisher(
            PointCloud2,
            args.topic,
            qos,
        )

        self.get_logger().info(
            f"Cloud ready: {len(xyz):,} points"
        )

        self.get_logger().info(
            f"Topic: {args.topic}"
        )

        self.get_logger().info(
            f"Frame: {args.frame}"
        )

        self.get_logger().info(
            f"Message size: "
            f"{len(self.cloud.data) / 1024**2:.2f} MiB"
        )

        # Publish once and retain the message for late subscribers.
        # Keep this node alive so transient-local data remains available.
        self.timer = self.create_timer(
            1.0,
            self.publish_once,
        )

    def publish_once(self):
        self.cloud.header.stamp = (
            self.get_clock().now().to_msg()
        )

        self.publisher.publish(self.cloud)

        self.get_logger().info(
            "Point cloud published successfully"
        )

        self.timer.cancel()


def main():

    parser = argparse.ArgumentParser(
        description="Publish a colored point cloud in ROS 2"
    )

    parser.add_argument(
        "file",
        type=Path,
        help="Path to the .npz point cloud",
    )

    parser.add_argument(
        "--topic",
        default="/valmet/colored_cloud",
        help="ROS topic name",
    )

    parser.add_argument(
        "--frame",
        default="map",
        help="ROS coordinate frame",
    )

    args = parser.parse_args()

    if not args.file.exists():
        parser.error(
            f"File does not exist: {args.file}"
        )

    rclpy.init()

    node = ColoredCloudPublisher(args)

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()