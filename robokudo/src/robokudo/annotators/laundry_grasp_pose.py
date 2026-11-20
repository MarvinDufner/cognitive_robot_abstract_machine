"""
The grasp on the point of the laundry in a detected laundry basket that lies highest
above the floor.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import open3d
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
from robokudo.annotators.exceptions import (
    CameraImageAxisVertical,
    FloorPlaneMissing,
    GripperOpeningMissing,
)
from robokudo.cas import CAS, CASViews
from robokudo.types.annotation import Plane
from robokudo.types.camera import RegisteredDepthImage
from robokudo.types.cv import Point2D
from robokudo.types.laundry import (
    BasketFootprint,
    LaundryBasketClassName,
    LaundryClassName,
    LaundryGraspAnnotation,
    LaundrySurfaceShape,
)
from robokudo.types.scene import ObjectHypothesis
from semantic_digital_twin.world_description.geometry import Color

# %% annotator


class LaundryGraspPoseAnnotator(BaseAnnotator):
    """
    Annotates each laundry basket with a top-down grasp on the highest wide, steady top
    of its laundry that keeps the open fingers clear of the walls, closing across the
    ridge there, and shows the laundry as a heatmap of its height above the floor.
    """

    @classproperty
    def height_colormap(cls) -> int:
        """
        The OpenCV colormap the heatmap colors the laundry with, from its lowest to its
        highest point.
        """
        return cv2.COLORMAP_TURBO

    @classproperty
    def grasp_marker_color(cls) -> Color:
        """
        The color the grasp is marked in, in the image and in the point cloud.
        """
        return Color.from_rgba(red=1.0, green=0.0, blue=0.0)

    @classproperty
    def grasp_arrow_length(cls) -> float:
        """
        The length of the arrow marking the grasp in the point cloud, in metres.
        """
        return 0.1

    @classproperty
    def grasp_arrow_radius(cls) -> float:
        """
        The radius of the arrow's shaft, in metres.
        """
        return 0.004

    @classproperty
    def grasp_arrow_head_length(cls) -> float:
        """
        The length of the arrow's head, in metres.
        """
        return 0.025

    @classproperty
    def grasp_arrow_head_radius(cls) -> float:
        """
        The radius of the arrow's head at its base, in metres.
        """
        return 0.01

    @classproperty
    def centimetres_per_metre(cls) -> float:
        """
        How many centimetres, in which the heatmap's scale is labeled, make up a metre.
        """
        return 100.0

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration of the laundry grasp.
        """

        class Parameters:
            """
            Parameters of the laundry grasp.
            """

            def __init__(self) -> None:
                """
                Set the default parameters of the laundry grasp.
                """
                self.laundry_edge_width: int = 5
                """
                How many pixels at the edge of the measured laundry the grasp leaves
                out, so that segmentation spilling onto the basket's rim and depth
                measured across the rim's edge cannot place the grasp there.
                """
                self.grasp_area_radius: float = 0.05
                """
                The horizontal radius around the grasp point, in metres, of the laundry
                whose shape decides the direction the fingers close along.
                """
                self.ridge_dominance: float = 2.0
                """
                How many times as strongly the laundry has to curve down across a ridge
                as along it for the fingers to close across the ridge.
                """
                self.gripper_maximum_opening: float | None = None
                """
                The widest the robot's gripper opens, in metres. The grasp keeps half
                of it between the grasp point and the basket's walls, so that the open
                fingers clear them.
                """
                self.maximum_slope: float = 2.0
                """
                How steeply, as rise over run, the measured heights within
                :attr:`slope_window` pixels of a pixel may rise for it to be grasped.
                A steeper rise marks a sharp edge, such as a wall's, or depth measured
                across one.
                """
                self.slope_window: int = 2
                """
                How many pixels around a pixel its slope is measured across.
                """
                self.minimum_top_width: float = 0.01
                """
                How wide, in metres, the laundry has to stand at least as high as the
                grasp point all around it, so that a single noisy pixel or a thin edge
                cannot hold the grasp.
                """
                self.maximum_curvature: float = 60.0
                """
                How sharply, in 1/m, the laundry may curve down around the grasp point.
                No grasp is taken where it curves down more sharply, as cloth does not.
                """

        parameters = Parameters()

    def __init__(
        self,
        name: str = "LaundryGraspPoseAnnotator",
        descriptor: LaundryGraspPoseAnnotator.Descriptor | None = None,
    ) -> None:
        """
        :param name: The name of the node in the tree.
        :param descriptor: The configuration of the laundry grasp.
        """
        super().__init__(name, descriptor)

    def update(self) -> Status:
        """
        Annotate each laundry basket with the grasp on its laundry, where the camera
        measured depth on it, and show the laundry's heights above the floor.

        :raises GripperOpeningMissing: If no maximum opening of the gripper is set.
        :raises FloorPlaneMissing: If no plane was detected to measure heights from.
        :raises CameraImageAxisVertical: If the image's horizontal axis points straight
            up or down in the world.
        """
        if self.descriptor.parameters.gripper_maximum_opening is None:
            raise GripperOpeningMissing()
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
        slopes = self.slope_image(
            depth=depth, floor=floors[-1], image_shape=image_shape
        )
        laundry_heights = np.full(image_shape, np.nan)
        grasps: list[LaundryGraspAnnotation] = []
        for hypothesis in hypotheses:
            if (
                not LaundryBasketClassName.classifies(hypothesis)
                or hypothesis.roi.mask is None
            ):
                continue
            heights = self.laundry_height_image(
                basket=hypothesis,
                laundry=laundry,
                depth=depth,
                floor=floors[-1],
                image_shape=image_shape,
            )
            laundry_heights = np.fmax(laundry_heights, heights)
            grasp = self.laundry_grasp(
                basket=hypothesis,
                heights=heights,
                slopes=slopes,
                depth=depth,
            )
            if grasp is None:
                continue
            hypothesis.annotations.append(grasp)
            grasps.append(grasp)
        self.draw_height_heatmap(visualization_image, laundry_heights)
        self.draw_grasp_markers(
            image=visualization_image, pixels=[grasp.pixel for grasp in grasps]
        )
        self.get_annotator_output_struct().set_image(visualization_image)
        self.show_grasp_arrows_in_cloud(cas=cas, grasps=grasps, depth=depth)
        return Status.SUCCESS

    def laundry_grasp(
        self,
        *,
        basket: ObjectHypothesis,
        heights: npt.NDArray[np.float64],
        slopes: npt.NDArray[np.float32],
        depth: RegisteredDepthImage,
    ) -> LaundryGraspAnnotation | None:
        """
        :param basket: A detected laundry basket with a mask.
        :param heights: The height of the basket's laundry at each pixel, ``NaN``
            elsewhere.
        :param slopes: How steeply the heights around each pixel rise.
        :param depth: The depth image the laundry was measured in.
        :return: The grasp on the highest graspable top of the laundry, or ``None`` if
            the laundry offers none or curves down there more sharply than cloth.
        """
        graspable_heights = self.graspable_heights(
            basket=basket, heights=heights, slopes=slopes, depth=depth
        )
        grasp_pixel = self.top_pixel(heights=graspable_heights, depth=depth)
        if grasp_pixel is None:
            self.rk_logger.warning("No graspable laundry detected in %s", basket)
            return None
        grasp_point = depth.world_point(grasp_pixel)
        surface = self.surface_shape(
            grasp_point=grasp_point, heights=graspable_heights, depth=depth
        )
        if (
            surface is not None
            and surface.across_curvature < -self.descriptor.parameters.maximum_curvature
        ):
            self.rk_logger.warning(
                "The top in %s curves down more sharply than cloth", basket
            )
            return None
        return LaundryGraspAnnotation(
            source=self.name,
            pixel=grasp_pixel,
            height_above_floor=float(heights[grasp_pixel.y, grasp_pixel.x]),
            grasp_pose=self.aligned_grasp_pose(
                grasp_point=grasp_point, surface=surface, depth=depth
            ),
        )

    def graspable_heights(
        self,
        *,
        basket: ObjectHypothesis,
        heights: npt.NDArray[np.float64],
        slopes: npt.NDArray[np.float32],
        depth: RegisteredDepthImage,
    ) -> npt.NDArray[np.float64]:
        """
        :param basket: A detected laundry basket with a mask.
        :param heights: The height of the basket's laundry at each pixel, ``NaN``
            elsewhere.
        :param slopes: How steeply the heights around each pixel rise.
        :param depth: The depth image the laundry and the basket were measured in.
        :return: The heights, ``NaN`` also at the laundry's edge, where the heights
            rise more steeply than :attr:`Descriptor.Parameters.maximum_slope` and
            closer to the basket's walls than half the gripper's opening.
        """
        parameters = self.descriptor.parameters
        steady_heights = np.where(
            slopes <= parameters.maximum_slope,
            self.heights_within_edge(heights=heights),
            np.nan,
        )
        graspable_heights = np.full(heights.shape, np.nan)
        basket_rows, basket_columns = np.nonzero(
            basket.roi.full_image_mask(heights.shape)
        )
        basket_points = depth.world_points(
            np.column_stack([basket_columns, basket_rows])
        )
        basket_points = basket_points[~np.isnan(basket_points).any(axis=1)]
        if len(basket_points) == 0:
            return graspable_heights
        rows, columns = np.nonzero(~np.isnan(steady_heights))
        clearances = BasketFootprint.from_points(points=basket_points).distances_inside(
            points=depth.world_points(np.column_stack([columns, rows]))
        )
        clear = clearances >= parameters.gripper_maximum_opening / 2
        graspable_heights[rows[clear], columns[clear]] = steady_heights[
            rows[clear], columns[clear]
        ]
        return graspable_heights

    def slope_image(
        self,
        *,
        depth: RegisteredDepthImage,
        floor: Plane,
        image_shape: tuple[int, int],
    ) -> npt.NDArray[np.float32]:
        """
        :param depth: The depth image the heights are measured from.
        :param floor: The floor plane, in the optical frame.
        :param image_shape: The height and width of the color image.
        :return: How steeply, as rise over run, the heights above the floor measured
            within :attr:`Descriptor.Parameters.slope_window` pixels of each color image
            pixel rise, ``NaN`` where the camera measured no depth.
        """
        rows, columns = np.indices(image_shape).reshape(2, -1)
        camera_points = depth.camera_points(np.column_stack([columns, rows]))
        scene_heights = floor.heights(camera_points).reshape(image_shape)
        measured = ~np.isnan(scene_heights)
        window = self.descriptor.parameters.slope_window
        kernel = np.ones((2 * window + 1, 2 * window + 1), dtype=np.uint8)
        highest = cv2.dilate(
            np.where(measured, scene_heights, -np.inf).astype(np.float32), kernel
        )
        lowest = cv2.erode(
            np.where(measured, scene_heights, np.inf).astype(np.float32), kernel
        )
        runs = (2 * window * camera_points[:, 2] / depth.color_focal_length).reshape(
            image_shape
        )
        return (highest - lowest) / runs

    def top_pixel(
        self, *, heights: npt.NDArray[np.float64], depth: RegisteredDepthImage
    ) -> Point2D | None:
        """
        :param heights: The height of the laundry at each pixel, ``NaN`` where it may
            not be grasped.
        :param depth: The depth image the laundry was measured in.
        :return: The center of the highest top around which the laundry stands at
            least as high across :attr:`Descriptor.Parameters.minimum_top_width`, or
            ``None`` if the laundry has no such top.
        """
        measured = ~np.isnan(heights)
        if not measured.any():
            return None
        rows, columns = np.nonzero(measured)
        distance = np.nanmedian(
            depth.camera_points(np.column_stack([columns, rows]))[:, 2]
        )
        half_width = int(
            np.ceil(
                self.descriptor.parameters.minimum_top_width
                * depth.color_focal_length
                / distance
                / 2
            )
        )
        kernel = np.ones((2 * half_width + 1, 2 * half_width + 1), dtype=np.uint8)
        top_heights = cv2.erode(
            np.where(measured, heights, -np.inf).astype(np.float32), kernel
        )
        if not np.isfinite(top_heights).any():
            return None
        row, column = np.unravel_index(np.argmax(top_heights), heights.shape)
        return Point2D(x=int(column), y=int(row))

    def draw_grasp_markers(
        self, *, image: npt.NDArray[np.uint8], pixels: list[Point2D]
    ) -> None:
        """
        Draw a dot in :attr:`grasp_marker_color`, outlined in black, at each grasp.

        :param image: The image to draw on, in BGR order.
        :param pixels: The color image pixels of the grasps.
        """
        red, green, blue = self.grasp_marker_color.to_rgb()
        marker_bgr = tuple(
            round(channel * np.iinfo(np.uint8).max) for channel in (blue, green, red)
        )
        for pixel in pixels:
            cv2.circle(image, (pixel.x, pixel.y), 7, (0, 0, 0), -1)
            cv2.circle(image, (pixel.x, pixel.y), 5, marker_bgr, -1)

    def show_grasp_arrows_in_cloud(
        self,
        *,
        cas: CAS,
        grasps: list[LaundryGraspAnnotation],
        depth: RegisteredDepthImage,
    ) -> None:
        """
        Show an arrow in :attr:`grasp_marker_color` along each grasp's approach, its tip
        at the grasp, within the scene's point cloud if the CAS holds one.

        :param cas: The CAS that may hold the point cloud, in the optical frame.
        :param grasps: The grasps found in the frame.
        :param depth: The depth image whose camera pose relates the world frame to the
            optical frame.
        """
        geometries = []
        if cas.contains(CASViews.CLOUD):
            geometries.append({"name": "Cloud", "geometry": cas.get(CASViews.CLOUD)})
        camera_T_world = np.linalg.inv(depth.world_T_camera.to_np())
        for index, grasp in enumerate(grasps):
            camera_T_grasp = camera_T_world @ grasp.grasp_pose.to_np()
            approach, closing, normal, tip = camera_T_grasp[:3].T
            arrow = open3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=self.grasp_arrow_radius,
                cone_radius=self.grasp_arrow_head_radius,
                cylinder_height=self.grasp_arrow_length - self.grasp_arrow_head_length,
                cone_height=self.grasp_arrow_head_length,
            )
            camera_T_arrow = np.eye(4)
            camera_T_arrow[:3, :3] = np.column_stack([closing, normal, approach])
            camera_T_arrow[:3, 3] = tip - self.grasp_arrow_length * approach
            arrow.transform(camera_T_arrow)
            arrow.paint_uniform_color(self.grasp_marker_color.to_rgb())
            geometries.append({"name": f"Grasp {index}", "geometry": arrow})
        self.get_annotator_output_struct().set_geometries(geometries)

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

    def heights_within_edge(
        self, *, heights: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        :param heights: The height of the laundry at each pixel, ``NaN`` where none was
            measured.
        :return: The heights, ``NaN`` also within :attr:`Descriptor.Parameters.laundry_edge_width`
            pixels of the edge of the measured laundry.
        """
        edge_width = self.descriptor.parameters.laundry_edge_width
        edge_kernel = np.ones((2 * edge_width + 1, 2 * edge_width + 1), dtype=np.uint8)
        measured = (~np.isnan(heights)).astype(np.uint8)
        inside_edge = cv2.erode(measured, edge_kernel).astype(bool)
        return np.where(inside_edge, heights, np.nan)

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

    def aligned_grasp_pose(
        self,
        *,
        grasp_point: Point3,
        surface: LaundrySurfaceShape | None,
        depth: RegisteredDepthImage,
    ) -> Pose:
        """
        :param grasp_point: Where to take hold, in the world frame.
        :param surface: The shape of the laundry around the grasp point, ``None`` if
            too little of it was measured.
        :param depth: The depth image the laundry was measured in.
        :return: A grasp at the point approaching straight down along its x-axis and
            closing along its y-axis across the ridge the laundry forms there, or along
            the image's horizontal axis laid flat where no ridge stands out.
        :raises CameraImageAxisVertical: If the image's horizontal axis points straight
            up or down in the world.
        """
        image_axis_closing = self.image_axis_closing(depth=depth)
        if surface is None or not surface.has_ridge(
            dominance=self.descriptor.parameters.ridge_dominance
        ):
            return self.top_down_grasp_pose(
                grasp_point=grasp_point, closing=image_axis_closing, depth=depth
            )
        ridge_closing = surface.across_direction
        # The fingers are symmetric; keep the one closest to the image axis so that
        # the wrist does not flip half a turn between frames.
        if ridge_closing @ image_axis_closing < 0.0:
            ridge_closing = -ridge_closing
        return self.top_down_grasp_pose(
            grasp_point=grasp_point, closing=ridge_closing, depth=depth
        )

    def surface_shape(
        self,
        *,
        grasp_point: Point3,
        heights: npt.NDArray[np.float64],
        depth: RegisteredDepthImage,
    ) -> LaundrySurfaceShape | None:
        """
        :param grasp_point: Where to take hold, in the world frame.
        :param heights: The height of the laundry at each pixel, ``NaN`` where it may
            not be grasped.
        :param depth: The depth image the laundry was measured in.
        :return: The shape of the laundry within
            :attr:`Descriptor.Parameters.grasp_area_radius` of the grasp, or ``None``
            if too little of it was measured to fit one.
        """
        rows, columns = np.nonzero(~np.isnan(heights))
        offsets = (
            depth.world_points(np.column_stack([columns, rows]))
            - grasp_point.to_np()[:3]
        )
        offsets = offsets[~np.isnan(offsets).any(axis=1)]
        offsets = offsets[
            np.linalg.norm(offsets[:, :2], axis=1)
            <= self.descriptor.parameters.grasp_area_radius
        ]
        x, y, z = offsets.T
        quadratic_terms = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
        if len(offsets) < quadratic_terms.shape[1]:
            return None
        (xx, xy, yy, *_), *_ = np.linalg.lstsq(quadratic_terms, z, rcond=None)
        curvatures, directions = np.linalg.eigh(
            np.array([[2.0 * xx, xy], [xy, 2.0 * yy]])
        )
        return LaundrySurfaceShape(
            across_curvature=float(curvatures[0]),
            along_curvature=float(curvatures[1]),
            across_direction=np.array([*directions[:, 0], 0.0]),
        )

    @staticmethod
    def image_axis_closing(*, depth: RegisteredDepthImage) -> npt.NDArray[np.float64]:
        """
        :param depth: The depth image whose camera pose gives the image's axes.
        :return: The image's horizontal axis laid flat in the world, of unit length.
        :raises CameraImageAxisVertical: If the image's horizontal axis points straight
            up or down in the world.
        """
        image_x_axis: npt.NDArray[np.float64] = depth.world_T_camera.to_np()[:3, 0]
        flat_image_x_axis = np.array([image_x_axis[0], image_x_axis[1], 0.0])
        length = np.linalg.norm(flat_image_x_axis)
        if np.isclose(length, 0.0):
            raise CameraImageAxisVertical()
        return flat_image_x_axis / length

    @staticmethod
    def top_down_grasp_pose(
        *,
        grasp_point: Point3,
        closing: npt.NDArray[np.float64],
        depth: RegisteredDepthImage,
    ) -> Pose:
        """
        :param grasp_point: Where to take hold, in the world frame.
        :param closing: The flat world direction the fingers close along.
        :param depth: The depth image whose world frame the grasp is given in.
        :return: A grasp at the point approaching straight down along its x-axis and
            closing along its y-axis.
        """
        frame = depth.world_T_camera.reference_frame
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
