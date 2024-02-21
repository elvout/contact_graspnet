from dataclasses import dataclass, field, is_dataclass
from typing import Any

from rclpy.node import Node


@dataclass
class ContactGraspnetParams:
    """
    NOTE: This is a subset of parameters specific to the contact_graspnet
    library.
    """

    ckpt_dir: str = field(default="checkpoints/scene_test_2048_bs3_hor_sigma_001")

    z_range: list[float] = field(default_factory=lambda: [0.2, 1.8])
    """Z value threshold to crop the input point cloud."""

    local_regions: bool = field(default=False)
    """Whether to crop 3D local regions around given segments."""

    filter_grasps: bool = field(default=False)
    """Whether to filter grasp contacts according to segmentation maps."""

    skip_border_objects: bool = field(default=False)
    """Whether to ignore segments at the depth map boundary when extracting
    local regions."""

    forward_passes: int = field(default=1)
    """Number of parallel forward passes to run in mesh_utils for more potential
    contact points."""

    segmap_id: int = field(default=0)
    """Only return grasps with the given segment id (0 = all)."""

    visualize_grasps: bool = field(default=False)
    """Whether to generate grasp visualization images."""


@dataclass
class OneFormerParams:
    """
    NOTE: This is a subset of parameters specific to the OneFormer library.
    """

    pretrained_model_name: str = field(default="shi-labs/oneformer_ade20k_swin_large")
    """
    List of model names: https://huggingface.co/models?other=oneformer
    """

    visualize_segmentation: bool = field(default=False)
    """Whether or not to generate visualization images."""

    stretch_robot_rotate_image_90deg: bool = field(default=False)
    """The Stretch robot's camera is mounted at a 90deg angle. This perspective
    is generally incompatible with OneFormer."""


@dataclass
class ContactGraspnetNodeParams:
    rgb_sub_topic: str = field(default="/camera/color/image_raw/compressed")
    """sensor_msgs/CompressedImage RGB topic to subscribe to."""

    depth_sub_topic: str = field(default="/camera/aligned_depth_to_color/image_raw")
    """
    sensor_msgs/Image aligned depth topic to subscribe to.

    Images published to this topic must share the same intrinsic matrix as the
    RGB image topic.

    The uncompressed topic is used as there seems to be some difficulty
    subscribing or decoding the compressed depth topic.
    """

    camera_info_sub_topic: str = field(default="/camera/color/camera_info")
    """sensor_msgs/CameraInfo topic to subscribe to."""

    contact_graspnet: ContactGraspnetParams = field(
        default_factory=ContactGraspnetParams
    )
    """
    NOTE: This field becomes a namespace of the same name in the parameter
    system. (i.e., contact_graspnet.foo )
    """

    oneformer: OneFormerParams = field(default_factory=OneFormerParams)
    """
    NOTE: This field becomes a namespace of the same name in the parameter
    system. (i.e., oneformer.foo )
    """

    @staticmethod
    def read_from_ros(node: Node) -> "ContactGraspnetNodeParams":
        params = ContactGraspnetNodeParams()

        def read_single_parameter(param_name: str, default_value: Any) -> Any:
            ros_value = node.declare_parameter(param_name, default_value).value
            assert type(ros_value) == type(default_value)
            return ros_value

        for field_name, field_value in vars(params).items():
            # TODO: this only handles one namespace level
            if is_dataclass(field_value):
                namespace_name: str = field_name
                namespace_params = field_value

                for param_name, param_default_value in vars(namespace_params).items():
                    namespaced_param_name = f"{namespace_name}.{param_name}"
                    setattr(
                        namespace_params,
                        param_name,
                        read_single_parameter(
                            namespaced_param_name, param_default_value
                        ),
                    )

                setattr(params, namespace_name, namespace_params)
            else:
                setattr(
                    params, field_name, read_single_parameter(field_name, field_value)
                )

        return params
