import sys
import threading
from typing import Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

import rclpy
from builtin_interfaces.msg import Duration, Time
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from visualization_msgs.msg import Marker, MarkerArray

from rosnode.contact_graspnet_wrapper import ContactGraspnetWrapper
from rosnode.oneformer_wrapper import OneFormerWrapper
from rosnode.parameters import ContactGraspnetNodeParams

class ColorGenerator:
    def __init__(self) -> None:
        # https://matplotlib.org/stable/gallery/color/colormap_reference.html
        cmap_name = "Set1"

        self._colors: Sequence[tuple[float, float, float]] = plt.get_cmap(
            cmap_name
        ).colors
        self._idx = 0

    def get_next_color(self) -> tuple[float, float, float]:
        color = self._colors[self._idx]

        self._idx += 1
        if self._idx >= len(self._colors):
            self._idx = 0

        return color


class ContactGraspnetNode(Node):
    def __init__(self) -> None:
        super().__init__("contact_graspnet")

        self._params = ContactGraspnetNodeParams.read_from_ros(self)

        self._contact_graspnet = ContactGraspnetWrapper(self._params.contact_graspnet)
        self._oneformer = OneFormerWrapper(self._params.oneformer)

        #####################
        # ROS-related setup #
        #####################
        self._marker_pub = self.create_publisher(
            Marker, "/contact_graspnet/best_grasp", 1
        )
        self._marker_array_pub = self.create_publisher(
            MarkerArray, "/contact_graspnet/all_grasps", 1
        )
        self._debug_img_pub = self.create_publisher(
            CompressedImage, "/contact_graspnet/vis/compressed", 1
        )

        # Data usd by subscribers and callbacks
        self._msg_lock = threading.Lock()
        self._intrinsic_matrix: Optional[np.ndarray] = None
        self._rgb_msg: Optional[CompressedImage] = None
        self._depth_msg: Optional[Image] = None
        self._cv_bridge = CvBridge()

        # TODO(elvout): how to we make sure these messages are synchronized?
        # Maybe track timestamps and only use timestamps with both messages
        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            self._params.camera_info_sub_topic,
            self._camera_info_callback,
            1,
        )
        self._rgb_sub = self.create_subscription(
            CompressedImage, self._params.rgb_sub_topic, self._rgb_callback, 1
        )
        self._depth_sub = self.create_subscription(
            Image, self._params.depth_sub_topic, self._depth_callback, 1
        )

        # TODO(elvout): switch to ros service
        self._inference_timer = self.create_timer(
            1 / 10, self.run_inference_and_publish_poses
        )

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        with self._msg_lock:
            self._intrinsic_matrix = np.array(msg.k).reshape((3, 3))

        # Assuming that the intrinsic matrix does not change, we only need to
        # receive one CameraInfo message. In ROS 1, we used
        # rospy.wait_for_message to avoid keeping track of a Subscriber object.
        # ROS 2 Humble does not have this feature (although it has since been
        # introduced https://github.com/ros2/rclpy/pull/960), so we manually
        # destroy the Subscription after receiving the CameraInfo message.
        self.destroy_subscription(self._camera_info_sub)

    def _rgb_callback(self, msg: CompressedImage) -> None:
        with self._msg_lock:
            self._rgb_msg = msg

    def _depth_callback(self, msg: Image) -> None:
        with self._msg_lock:
            self._depth_msg = msg

    def _grasp_pose_to_marker(self, pose: np.ndarray) -> Marker:
        """The caller is responsible for setting the frame_id, namespace, and
        id, and color."""

        # The direction of an arrow in rviz is given by +x axis of the
        # marker pose orientation, but the convention for gripper frames
        # has the "forward" direction as +z. We will perform an
        # intrinsic pitch adjustment of the basis vectors (right
        # multiplication) to align the +z axis of the gripper pose to
        # the +x axis of the visualization arrow.
        vis_pitch_correction = Rotation.from_euler("y", -np.pi / 2)

        pose = pose.copy()
        pose[:3, :3] = pose[:3, :3] @ vis_pitch_correction.as_matrix()
        # xyzw order
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()

        marker = Marker()
        marker.header.stamp = Time()
        marker.lifetime = Duration(sec=1, nanosec=int(0.5 * 1e9))
        marker.type = Marker.ARROW

        marker.scale.x = 0.08  # gripper length
        marker.scale.y = 0.015
        marker.scale.z = 0.015

        marker.color.a = 1.0

        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]

        marker.pose.position.x = pose[0, 3]
        marker.pose.position.y = pose[1, 3]
        marker.pose.position.z = pose[2, 3]

        return marker

    def run_inference_and_publish_poses(self) -> None:
        with self._msg_lock:
            if (
                self._intrinsic_matrix is None
                or self._rgb_msg is None
                or self._depth_msg is None
            ):
                return

            rgb_msg_header = self._rgb_msg.header

            bgr_image = self._cv_bridge.compressed_imgmsg_to_cv2(self._rgb_msg)
            depth_image = (
                self._cv_bridge.imgmsg_to_cv2(self._depth_msg).astype(np.float32)
                / 1000.0
            )
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            # Discard existing messages so that inference is only run again once
            # we have new data.
            self._rgb_msg = None
            self._depth_msg = None

        oneformer_output = self._oneformer.forward(rgb_image)

        graspnet_output = self._contact_graspnet.forward(
            rgb_image,
            depth_image,
            self._intrinsic_matrix,
            segmap_image=oneformer_output.segmap_image,
        )

        marker_array_msg = MarkerArray()
        color_generator = ColorGenerator()
        for segment_info in oneformer_output.segments_info:
            if segment_info.id not in graspnet_output.pred_grasps_cam.keys():
                continue

            if self._oneformer.label_id_to_str(segment_info.label_id) != "box":
                continue

            color = color_generator.get_next_color()
            local_grasps = graspnet_output.pred_grasps_cam[segment_info.id]
            for grasp_idx in range(local_grasps.shape[0]):
                grasp = local_grasps[grasp_idx]

                marker_msg = self._grasp_pose_to_marker(grasp)
                marker_msg.header.frame_id = rgb_msg_header.frame_id
                marker_msg.ns = f"{segment_info.id}-THESE-LABELS-ARE-NOT-CONSISTENT"
                marker_msg.id = grasp_idx

                marker_msg.color.r = color[0]
                marker_msg.color.g = color[1]
                marker_msg.color.b = color[2]

                marker_array_msg.markers.append(marker_msg)

        # self._marker_pub.publish(marker_msg)
        self._marker_array_pub.publish(marker_array_msg)

        if oneformer_output.vis_image is not None:
            debug_img_msg = self._cv_bridge.cv2_to_compressed_imgmsg(
                oneformer_output.vis_image, "png"
            )
            self._debug_img_pub.publish(debug_img_msg)


def main() -> None:
    rclpy.init(args=sys.argv)

    logger.info("Starting node initialization...")
    node = ContactGraspnetNode()
    logger.success("Node initialized.")

    rclpy.spin(node)


if __name__ == "__main__":
    main()
