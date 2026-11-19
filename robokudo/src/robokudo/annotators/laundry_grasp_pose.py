"""
The grasp on the point of the laundry in a detected laundry basket that lies highest
above the floor.
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
from krrood.ormatic.utils import classproperty
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.exceptions import CameraImageAxisVertical, FloorPlaneMissing
from robokudo.cas import CAS, CASViews
from robokudo.types.annotation import Plane
from robokudo.types.camera import RegisteredDepthImage
from robokudo.types.cv import Point2D
from robokudo.types.laundry import (
    LaundryBasketClassName,
    LaundryClassName,
    LaundryGraspAnnotation,
)
from robokudo.types.scene import ObjectHypothesis

# %% annotator


class LaundryGraspPoseAnnotator(BaseAnnotator):
    """
    Annotates each laundry basket with a grasp on the point of the segmented laundry
    detected in it that lies highest above the floor, so that a garment can be lifted
    out, and shows the laundry as a heatmap of its height above the floor.

    The grasp approaches straight down and closes along the image's horizontal axis, as
    seen in the world.
    """

    @classproperty
    def height_colormap(cls) -> int:
        """
        The OpenCV colormap the heatmap colors the laundry with, from its lowest to its
        highest point.
        """
        return cv2.COLORMAP_TURBO

    @classproperty
    def centimetres_per_metre(cls) -> float:
        """
        How many centimetres, in which the heatmap's scale is labeled, make up a metre.
        """
        return 100.0

    def __init__(
        self,
        name: str = "LaundryGraspPoseAnnotator",
        descriptor: BaseAnnotator.Descriptor | None = None,
    ) -> None:
        super().__init__(name, descriptor)

    def update(self) -> Status:
        """
        Annotate each laundry basket with the grasp on its laundry, where the camera
        measured depth on it, and show the laundry's heights above the floor.

        :raises FloorPlaneMissing: If no plane was detected to measure heights from.
        :raises CameraImageAxisVertical: If the image's horizontal axis points straight
            up or down in the world.
        """
        cas = self.get_cas()
        floors = CAS.filter_by_type(Plane, cas.annotations)
        if len(floors) == 0:
            raise FloorPlaneMissing()
        depth = RegisteredDepthImage.from_cas(cas)
        visualization_image = cas.get_copy(CASViews.COLOR_IMAGE)
        image_shape = visualization_image.shape[:2]
        hypotheses = cas.filter_annotations_by_type(ObjectHypothesis)
        laundry = [
            hypothesis
            for hypothesis in hypotheses
            if LaundryClassName.classifies(hypothesis)
            and hypothesis.roi.mask is not None
        ]
        laundry_heights = np.full(image_shape, np.nan)
        grasp_pixels: list[Point2D] = []
        for hypothesis in hypotheses:
            if not LaundryBasketClassName.classifies(hypothesis):
                continue
            heights = self.laundry_height_image(
                basket=hypothesis,
                laundry=laundry,
                depth=depth,
                floor=floors[-1],
                image_shape=image_shape,
            )
            if np.isnan(heights).all():
                self.rk_logger.warning(
                    "No laundry with measured depth detected in %s", hypothesis
                )
                continue
            row, column = np.unravel_index(np.nanargmax(heights), image_shape)
            highest_pixel = Point2D(x=int(column), y=int(row))
            hypothesis.annotations.append(
                LaundryGraspAnnotation(
                    source=self.name,
                    pixel=highest_pixel,
                    height_above_floor=float(heights[row, column]),
                    grasp_pose=self.top_down_grasp_pose(
                        depth.world_point(highest_pixel), depth
                    ),
                )
            )
            laundry_heights = np.fmax(laundry_heights, heights)
            grasp_pixels.append(highest_pixel)
        self.draw_height_heatmap(visualization_image, laundry_heights)
        for pixel in grasp_pixels:
            cv2.circle(visualization_image, (pixel.x, pixel.y), 6, (0, 0, 0), -1)
            cv2.circle(visualization_image, (pixel.x, pixel.y), 4, (255, 255, 255), -1)
        self.get_annotator_output_struct().set_image(visualization_image)
        return Status.SUCCESS

    @staticmethod
    def laundry_height_image(
        *,
        basket: ObjectHypothesis,
        laundry: list[ObjectHypothesis],
        depth: RegisteredDepthImage,
        floor: Plane,
        image_shape: tuple[int, int],
    ) -> npt.NDArray[np.float64]:
        """
        :param basket: A detected laundry basket.
        :param laundry: The segmented laundry detected anywhere in the image.
        :param depth: The depth image the heights are measured from.
        :param floor: The floor plane, in the optical frame.
        :param image_shape: The height and width of the color image.
        :return: The height above the floor of the laundry overlapping the basket's
            region at each color image pixel, ``NaN`` elsewhere and where the camera
            measured no depth.
        """
        left, top, right, bottom = basket.roi.roi.get_corner_points()
        laundry_mask = np.zeros(image_shape, dtype=np.uint8)
        for garment in laundry:
            garment_mask = garment.roi.full_image_mask(image_shape)
            if garment_mask[top:bottom, left:right].any():
                laundry_mask |= garment_mask
        rows, columns = np.nonzero(laundry_mask)
        heights = np.full(image_shape, np.nan)
        heights[rows, columns] = floor.heights(
            depth.camera_points(np.column_stack([columns, rows]))
        )
        return heights

    def draw_height_heatmap(
        self, image: npt.NDArray[np.uint8], heights: npt.NDArray[np.float64]
    ) -> None:
        """
        Color each pixel with a measured height by :attr:`height_colormap`, from the
        lowest to the highest height, and draw the scale in centimetres at the image's
        right edge.

        :param image: The image to draw on, in BGR order.
        :param heights: The height above the floor at each pixel of the image.
        """
        measured = ~np.isnan(heights)
        if not measured.any():
            return
        lowest, highest = heights[measured].min(), heights[measured].max()
        height_range = highest - lowest if highest > lowest else 1.0
        levels = np.round(
            (heights[measured] - lowest) / height_range * np.iinfo(np.uint8).max
        ).astype(np.uint8)
        image[measured] = cv2.applyColorMap(levels[:, None], self.height_colormap)[:, 0]

        image_height, image_width = image.shape[:2]
        bar_top, bar_bottom = image_height // 4, image_height * 3 // 4
        bar_left, bar_right = image_width - 22, image_width - 10
        bar_levels = np.linspace(
            np.iinfo(np.uint8).max, 0, bar_bottom - bar_top
        ).astype(np.uint8)
        image[bar_top:bar_bottom, bar_left:bar_right] = cv2.applyColorMap(
            bar_levels[:, None], self.height_colormap
        )
        for height, row in ((highest, bar_top), (lowest, bar_bottom)):
            cv2.putText(
                image,
                f"{height * self.centimetres_per_metre:.1f} cm",
                (bar_left - 70, row + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

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
