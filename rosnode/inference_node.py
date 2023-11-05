"""
Usage: python3 -m rosnode.inference_node
"""
from typing import Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from visualization_msgs.msg import Marker, MarkerArray

from .contact_graspnet_wrapper import ContactGraspnetParams, ContactGraspnetWrapper
from .oneformer_wrapper import OneFormerParams, OneFormerWrapper


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


class ContactGraspnetInferencePubSub:
    def __init__(self) -> None:
        ##################################
        # Contact Graspnet-related setup #
        ##################################
        contact_graspnet_params = ContactGraspnetParams()
        contact_graspnet_params.ckpt_dir = rospy.get_param(
            "contact_graspnet_ckpt_dir",
            default="checkpoints/scene_test_2048_bs3_hor_sigma_001",
        )
        contact_graspnet_params.z_range[0] = rospy.get_param("z_min", default=0.2)
        contact_graspnet_params.z_range[1] = rospy.get_param("z_max", default=1.8)
        contact_graspnet_params.local_regions = rospy.get_param(
            "local_regions", default=True
        )
        contact_graspnet_params.filter_grasps = rospy.get_param(
            "filter_grasps", default=True
        )
        contact_graspnet_params.forward_passes = rospy.get_param(
            "forward_passes", default=1
        )
        contact_graspnet_params.skip_border_objects = rospy.get_param(
            "skip_border_objects", default=False
        )
        contact_graspnet_params.segmap_id = rospy.get_param("segmap_id", default=0)

        self._contact_graspnet = ContactGraspnetWrapper(contact_graspnet_params)

        ###########################
        # OneFormer-related setup #
        ###########################
        oneformer_params = OneFormerParams()
        oneformer_params.stretch_robot_rotate_image_90deg = rospy.get_param(
            "stretch_robot_rotate_image_90deg", default=True
        )
        self._oneformer = OneFormerWrapper(oneformer_params)

        #####################
        # ROS-related setup #
        #####################
        camera_info_topic: str = rospy.get_param(
            "camera_info_topic", default="/camera/aligned_depth_to_color/camera_info"
        )
        # Subscribe to the uncompressed image since there seems to be some
        # difficulty subscribing to the /compressedDepth topic.
        depth_topic: str = rospy.get_param(
            "depth_topic", default="/camera/aligned_depth_to_color/image_raw"
        )
        rgb_topic: str = rospy.get_param(
            "rgb_topic", default="/camera/color/image_raw/compressed"
        )

        # Perform a blocking read of a single CameraInfo message instead of
        # creating a persistent subscriber.
        logger.info("Waiting 5s for CameraInfo...")
        camera_info_msg: CameraInfo = rospy.wait_for_message(
            camera_info_topic, CameraInfo, timeout=5
        )
        self._intrinsic_matrix = np.array(camera_info_msg.K).reshape((3, 3))
        self._rgb_msg: Optional[CompressedImage] = None
        self._depth_msg: Optional[Image] = None
        self._cv_bridge = CvBridge()

        self._marker_pub = rospy.Publisher(
            "/contact_graspnet/best_grasp", Marker, queue_size=1
        )
        self._marker_array_pub = rospy.Publisher(
            "/contact_graspnet/all_grasps", MarkerArray, queue_size=1
        )
        self._debug_img_pub = rospy.Publisher(
            "/contact_graspnet/vis/compressed", CompressedImage, queue_size=1
        )

        # TODO(elvout): how to we make sure these messages are synchronized?
        # Maybe track timestamps and only use timestamps with both messages
        self._rgb_sub = rospy.Subscriber(
            rgb_topic, CompressedImage, self._rgb_callback, queue_size=1
        )
        self._depth_sub = rospy.Subscriber(
            depth_topic, Image, self._depth_callback, queue_size=1
        )

    def _rgb_callback(self, msg: CompressedImage) -> None:
        self._rgb_msg = msg

    def _depth_callback(self, msg: Image) -> None:
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
        marker.header.stamp = rospy.Time(0)
        marker.lifetime = rospy.Time.from_sec(1.5)
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
        if self._rgb_msg is None or self._depth_msg is None:
            return

        rgb_msg_header = self._rgb_msg.header
        bgr_image = self._cv_bridge.compressed_imgmsg_to_cv2(self._rgb_msg)
        depth_image = (
            self._cv_bridge.imgmsg_to_cv2(self._depth_msg).astype(np.float32) / 1000.0
        )
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

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
    logger.info("Starting node initialization...")
    rospy.init_node("contact_graspnet")
    pubsub = ContactGraspnetInferencePubSub()
    logger.success("Node initialized.")

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        pubsub.run_inference_and_publish_poses()
        rate.sleep()  # TODO(elvout): take into account inference latency?


if __name__ == "__main__":
    main()
