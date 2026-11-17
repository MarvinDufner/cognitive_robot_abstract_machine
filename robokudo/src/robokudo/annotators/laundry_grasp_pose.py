"""
The grasp on the highest point of the laundry in a detected laundry basket.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
from krrood_spatial_math.spatial_types import (
    Point3,
    Pose,
    Quaternion,
    RotationMatrix,
    Vector3,
)
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.exceptions import CameraImageAxisVertical
from robokudo.cas import CASViews
from robokudo.types.camera import RegisteredDepthImage
from robokudo.types.cv import Point2D
from robokudo.types.laundry import LaundryBasketClassName, LaundryGraspAnnotation
from robokudo.types.scene import ObjectHypothesis

# %% annotator


class LaundryGraspPoseAnnotator(BaseAnnotator):
    """
    Annotates each segmented laundry basket with a grasp on the highest point of the
    laundry inside it, leaving out the basket's rim along its outline.

    The grasp approaches straight down and closes along the image's horizontal axis, as
    seen in the world.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Descriptor for the LaundryGraspPoseAnnotator.
        """

        class Parameters:
            """
            Parameters for the LaundryGraspPoseAnnotator.
            """

            def __init__(self) -> None:
                self.rim_width_fraction: float = 0.15
                """
                How far in from the basket's outline its rim reaches, as a fraction of
                the shorter side of the basket's region of interest.
                """

    def __init__(
        self,
        name: str = "LaundryGraspPoseAnnotator",
        descriptor: LaundryGraspPoseAnnotator.Descriptor | None = None,
    ) -> None:
        super().__init__(name, descriptor)

    def update(self) -> Status:
        """
        Annotate each laundry basket with the grasp on its laundry, where the camera
        measured depth inside it.

        :raises CameraImageAxisVertical: If the image's horizontal axis points straight
            up or down in the world.
        """
        cas = self.get_cas()
        depth = RegisteredDepthImage.from_cas(cas)
        visualization_image = cas.get_copy(CASViews.COLOR_IMAGE)
        for hypothesis in cas.filter_annotations_by_type(ObjectHypothesis):
            if (
                not LaundryBasketClassName.is_laundry_basket(hypothesis)
                or hypothesis.roi.mask is None
            ):
                continue
            highest_pixel = self.highest_laundry_pixel(
                hypothesis, depth, visualization_image.shape[:2]
            )
            if highest_pixel is None:
                self.rk_logger.warning("No depth measured inside %s", hypothesis)
                continue
            hypothesis.annotations.append(
                LaundryGraspAnnotation(
                    source=self.name,
                    pixel=highest_pixel,
                    grasp_pose=self.top_down_grasp_pose(
                        depth.world_point(highest_pixel), depth
                    ),
                )
            )
            cv2.circle(
                visualization_image,
                (highest_pixel.x, highest_pixel.y),
                5,
                (0, 0, 255),
                -1,
            )
        self.get_annotator_output_struct().set_image(visualization_image)
        return Status.SUCCESS

    def highest_laundry_pixel(
        self,
        basket: ObjectHypothesis,
        depth: RegisteredDepthImage,
        image_shape: tuple[int, int],
    ) -> Point2D | None:
        """
        :param basket: A segmented laundry basket.
        :param depth: The depth image the heights are measured from.
        :param image_shape: The height and width of the color image.
        :return: The color image pixel at which the laundry inside the rim is highest,
            or ``None`` if the camera measured no depth there.
        """
        region = basket.roi.roi
        rim_width = round(
            self.descriptor.parameters.rim_width_fraction
            * min(region.width, region.height)
        )
        inside_rim = cv2.erode(
            basket.roi.full_image_mask(image_shape),
            np.ones((2 * rim_width + 1, 2 * rim_width + 1), np.uint8),
        )
        rows, columns = np.nonzero(inside_rim)
        pixels = np.column_stack([columns, rows])
        heights = depth.world_points(pixels)[:, 2]
        if np.isnan(heights).all():
            return None
        column, row = pixels[np.nanargmax(heights)]
        return Point2D(x=int(column), y=int(row))

    @staticmethod
    def top_down_grasp_pose(grasp_point: Point3, depth: RegisteredDepthImage) -> Pose:
        """
        :param grasp_point: Where to take hold, in the world frame.
        :param depth: The depth image whose camera pose decides the closing direction.
        :return: A grasp at the point approaching straight down along its x-axis and
            closing along its y-axis, the image's horizontal axis laid flat.
        :raises CameraImageAxisVertical: If the image's horizontal axis points straight
            up or down in the world.
        """
        frame = depth.world_T_camera.reference_frame
        image_x_axis: npt.NDArray[np.float64] = depth.world_T_camera.to_np()[:3, 0]
        flat_image_x_axis = np.array([image_x_axis[0], image_x_axis[1], 0.0])
        length = np.linalg.norm(flat_image_x_axis)
        if np.isclose(length, 0.0):
            raise CameraImageAxisVertical()
        closing = flat_image_x_axis / length
        rotation = RotationMatrix.from_vectors(
            x=Vector3(0.0, 0.0, -1.0, reference_frame=frame),
            y=Vector3(*closing, reference_frame=frame),
            reference_frame=frame,
        )
        return Pose(
            position=Point3(*grasp_point.to_np()[:3], reference_frame=frame),
            orientation=Quaternion.from_rotation_matrix(rotation),
            reference_frame=frame,
        )
