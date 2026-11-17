"""
The point cloud of each segmented object hypothesis, for detectors that only segment.
"""

from __future__ import annotations

from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.camera import DepthFrame
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image

# %% annotator


class HypothesisPointCloudAnnotator(BaseAnnotator):
    """
    Gives each object hypothesis that has a mask but no point cloud the colored point
    cloud of the depth measurements its mask covers, in the camera frame.
    """

    def __init__(
        self,
        name: str = "HypothesisPointCloudAnnotator",
        descriptor: BaseAnnotator.Descriptor | None = None,
    ) -> None:
        super().__init__(name, descriptor)

    def update(self) -> Status:
        """
        Build the point cloud of each segmented hypothesis that lacks one.
        """
        cas = self.get_cas()
        depth_frame = DepthFrame.from_cas(cas)
        color_image_shape = cas.get(CASViews.COLOR_IMAGE).shape[:2]
        for hypothesis in cas.filter_annotations_by_type(ObjectHypothesis):
            if hypothesis.points is not None or hypothesis.roi.mask is None:
                continue
            hypothesis.points = depth_frame.masked_cloud(
                mask=get_scaled_color_image_for_depth_image(
                    cas, hypothesis.roi.full_image_mask(color_image_shape)
                )
            )
        return Status.SUCCESS
