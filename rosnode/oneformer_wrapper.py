from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForUniversalSegmentation, AutoProcessor
from transformers.models.oneformer.processing_oneformer import OneFormerProcessor

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import cm


@dataclass
class OneFormerParams:
    pretrained_model_name: str = field(default="shi-labs/oneformer_ade20k_swin_large")
    """
    List of model names: https://huggingface.co/models?other=oneformer
    """

    visualize_segmentation: bool = field(default=False)
    """Whether or not to generate visualization images."""

    stretch_robot_rotate_image_90deg: bool = field(default=False)
    """The Stretch robot's camera is mounted at a 90deg angle. This perspective
    is generally incompatible with OneFormer."""


class OneFormerWrapper:
    @dataclass
    class SegmentInfo:
        id: int = field(default=0)
        label_id: int = field(default=0)
        was_fused: bool = field(default=False)
        score: float = field(default=0.0)

        def __init__(self, model_output: dict[str, Any]) -> None:
            self.id = model_output["id"]
            self.label_id = model_output["label_id"]
            self.was_fused = model_output["was_fused"]
            self.score = model_output["score"]

    @dataclass
    class ModelOutput:
        segmap_image: np.ndarray = field(
            default_factory=lambda: np.zeros((0, 0), dtype=np.uint8)
        )
        """HxW uint8"""

        segments_info: list["OneFormerWrapper.SegmentInfo"] = field(
            default_factory=list
        )

        vis_image: Optional[np.ndarray] = field(default=None)
        """HxWx3 uint8"""

    def __init__(self, params: OneFormerParams = OneFormerParams()) -> None:
        self._params = params

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            logger.warning("CUDA is not available to PyTorch!")
            self._device = torch.device("cpu")

        self._processor: OneFormerProcessor = AutoProcessor.from_pretrained(
            params.pretrained_model_name
        )
        self._model = AutoModelForUniversalSegmentation.from_pretrained(
            params.pretrained_model_name
        ).to(self._device)

    def label_id_to_str(self, label_id: int) -> str:
        return self._model.config.id2label[label_id]

    def _get_pretty_vis(
        self, segmentation: np.ndarray, segments_info: list[dict[str, Any]]
    ) -> np.ndarray:
        viridis = cm.get_cmap("viridis", np.max(segmentation))
        fig, ax = plt.subplots(layout="constrained")
        ax.imshow(segmentation)

        instances_counter: dict[str, int] = defaultdict(int)
        handles = []
        # for each segment, draw its legend
        for segment in segments_info:
            segment_id = segment["id"]
            segment_label_id = segment["label_id"]
            segment_label = self._model.config.id2label[segment_label_id]
            label = (
                f"[{segment_id}] {segment_label}-{instances_counter[segment_label_id]}"
            )
            instances_counter[segment_label_id] += 1
            color = viridis(segment_id)
            handles.append(mpatches.Patch(color=color, label=label))

        ax.legend(loc="upper right", handles=handles, bbox_to_anchor=(1.8, 1))

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        plt.close(fig)

        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return image

    def forward(self, rgb_image: np.ndarray) -> ModelOutput:
        """
        Arguments:
            rgb_image: HxWx3 uint8 image (RGB order)
        """
        if self._params.stretch_robot_rotate_image_90deg:
            rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)

        # Move channel axis to index 0
        rgb_image = np.moveaxis(rgb_image, -1, 0)

        panoptic_inputs = self._processor(
            images=rgb_image, task_inputs=["panoptic"], return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**panoptic_inputs)

        panoptic_segmentation = self._processor.post_process_panoptic_segmentation(
            outputs, label_ids_to_fuse=set(), target_sizes=[rgb_image.shape[1:]]
        )[0]

        # We'll assume that we have <= 254 masks (mask ids start at 1)
        segments_info: list[dict[str, Any]] = panoptic_segmentation["segments_info"]
        if len(segments_info) > 254:
            logger.warning("Found more than 254 objects: uint8 will overflow")
        segmentation: np.ndarray = (
            panoptic_segmentation["segmentation"].to("cpu").numpy().astype(np.uint8)
        )

        output = self.ModelOutput()

        if self._params.stretch_robot_rotate_image_90deg:
            output.segmap_image = cv2.rotate(
                segmentation, cv2.ROTATE_90_COUNTERCLOCKWISE
            )
        else:
            output.segmap_image = segmentation

        output.segments_info = [self.SegmentInfo(i) for i in segments_info]

        if self._params.visualize_segmentation:
            output.vis_image = self._get_pretty_vis(output.segmap_image, segments_info)

        return output
