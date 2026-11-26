"""
ROS-independent camera metadata for perception frames.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, replace

import cv2
import numpy as np
import open3d
from krrood.ormatic.utils import classproperty
from typing_extensions import TYPE_CHECKING

from robokudo.exceptions import CameraPoseMissing, InvalidCameraObservation
from robokudo.cas import CASViews
from robokudo.types.cv import ColorToDepthRatio, Point2D, Rect
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image
from robokudo.utils.o3d_helper import get_cloud_from_rgb_depth_and_mask
from robokudo.utils.ros_frames import normalize_ros_frame_id
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.robots.robot_parts import Camera
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    import numpy.typing as npt

    from robokudo.cas import CAS
    from semantic_digital_twin.datastructures.camera_resolution import CameraResolution

if TYPE_CHECKING:
    from semantic_digital_twin.datastructures.field_of_view import FieldOfView

# %% Camera definitions


@dataclass(frozen=True)
class CameraObservation:
    """
    Describe the camera state used to produce one perception frame.

    :attr:`world_T_camera` alone transforms optical-frame points into world coordinates.
    The camera root is the optical frame; its sampled pose is also its replay pose. A
    supplied pose must have a reference frame and identify this camera's optical root.
    Frame metadata and the camera root must remain unchanged after construction; use
    :meth:`with_world_T_camera` to replace the sampled pose.
    """

    camera: Camera
    """
    Semantic camera that produced the frame.
    """

    effective_camera_model: PinholeCameraModel
    """
    Effective projection model of the delivered image.
    """

    world_T_camera: HomogeneousTransformationMatrix | None
    """
    Sampled optical-frame pose in world coordinates, if available.

    This transform alone places optical-frame points in the world.
    """

    timestamp_nanoseconds: int
    """
    Acquisition time as nanoseconds since the Unix epoch.
    """

    def __post_init__(self) -> None:
        """
        Validate the sampled pose's reference frame and optical camera identity.

        :raises InvalidCameraObservation: If a supplied pose lacks a reference frame or
            identifies another camera.
        """
        pose = self.world_T_camera
        if pose is None:
            return
        if pose.reference_frame is None:
            raise InvalidCameraObservation(
                reason="sampled camera pose has no reference frame"
            )
        pose_child = self.pose_child
        camera_root = self.camera.root
        if pose_child is camera_root:
            return
        if (
            not isinstance(pose_child, Body)
            or pose_child.name.prefix != camera_root.name.prefix
            or pose_child.name.name == camera_root.name.name
            or normalize_ros_frame_id(pose_child.name.name)
            != normalize_ros_frame_id(camera_root.name.name)
        ):
            raise InvalidCameraObservation(
                reason="sampled pose child frame differs from the camera root"
            )

    @property
    def pose_child(self) -> Body:
        """
        Return the sampled child, or the camera root when no child is specified.

        ROS aliases differing only by leading slashes are accepted within the same name
        prefix. Validation occurs when the observation is constructed.

        :return: Optical camera root or its explicitly sampled ROS alias.
        """
        pose = self.world_T_camera
        if pose is None or pose.child_frame is None:
            return self.camera.root
        return pose.child_frame

    @property
    def resolution(self) -> CameraResolution:
        """
        Return the effective image resolution.
        """
        return self.effective_camera_model.resolution

    @property
    def field_of_view(self) -> FieldOfView:
        """
        Return the effective image field of view.
        """
        return self.effective_camera_model.field_of_view

    def world_T_camera_or_raise(self) -> HomogeneousTransformationMatrix:
        """
        Return the sampled pose or raise when the producer had none.

        :return: Transform from the optical frame into the world frame.
        :raises CameraPoseMissing: If the observation has no sampled pose.
        """
        if self.world_T_camera is None:
            raise CameraPoseMissing()
        return self.world_T_camera

    @property
    def camera_T_world(self) -> HomogeneousTransformationMatrix:
        """
        Return the inverse sampled camera pose.

        :raises CameraPoseMissing: If the observation has no sampled pose.
        """
        return self.world_T_camera_or_raise().inverse()

    def with_world_T_camera(
        self, world_T_camera: HomogeneousTransformationMatrix
    ) -> CameraObservation:
        """
        Return a copy carrying the supplied sampled camera pose.

        :param world_T_camera: Transform from the optical frame into the world frame.
        :return: Observation containing the supplied pose.
        :raises InvalidCameraObservation: If the pose lacks a reference frame or
            identifies another camera.
        """
        return replace(self, world_T_camera=world_T_camera)


# %% depth measurements


@dataclass
class RegisteredDepthImage:
    """
    A depth image together with what places its measurements in the world, so that a
    color image pixel can be looked up as the world point the camera saw there.

    .. note::
       Lens distortion is ignored.
    """

    depth_image: npt.NDArray
    """
    Distance of each depth image pixel along the optical axis, in millimetres, zero
    where the camera measured nothing.
    """

    camera_model: PinholeCameraModel
    """
    Projection model at the depth image resolution.
    """

    color_to_depth_ratio: ColorToDepthRatio
    """
    Size of the depth image relative to the color image.
    """

    world_T_camera: HomogeneousTransformationMatrix
    """
    Transform from the optical frame into the world frame.
    """

    @classproperty
    def millimetres_per_metre(cls) -> float:
        """
        How many of the depth image's units make up a metre.
        """
        return 1000.0

    @property
    def color_focal_length(self) -> float:
        """
        The horizontal focal length of the color image, in color image pixels.
        """
        return self.camera_model.focal_length_x / self.color_to_depth_ratio.horizontal

    @classmethod
    def from_cas(cls, cas: CAS) -> RegisteredDepthImage:
        """
        :param cas: A CAS holding a depth image and a camera observation with a pose.
        :return: The CAS's depth image, registered with its camera.
        :raises CameraObservationMissing: If the CAS has no camera observation.
        :raises CameraPoseMissing: If the observation has no sampled pose.
        """
        camera_observation = cas.require_camera_observation()
        color_to_depth_ratio = cas.color_to_depth_ratio
        return cls(
            depth_image=cas.depth_image,
            camera_model=camera_observation.effective_camera_model.scaled(
                scale_x=color_to_depth_ratio.horizontal,
                scale_y=color_to_depth_ratio.vertical,
            ),
            color_to_depth_ratio=color_to_depth_ratio,
            world_T_camera=camera_observation.world_T_camera_or_raise(),
        )

    def camera_points(
        self, color_pixels: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float64]:
        """
        :param color_pixels: The (x, y) of color image pixels, one pixel per row.
        :return: The (x, y, z) in the optical frame of the point the camera saw at each
            of those pixels, one point per row, ``NaN`` where it measured no depth or
            the pixel lies outside the image.
        """
        depth_pixels = np.floor(
            np.asarray(color_pixels, dtype=np.float64)
            * (self.color_to_depth_ratio.horizontal, self.color_to_depth_ratio.vertical)
        ).astype(int)
        height, width = self.depth_image.shape[:2]
        inside = (
            (depth_pixels[:, 0] >= 0)
            & (depth_pixels[:, 0] < width)
            & (depth_pixels[:, 1] >= 0)
            & (depth_pixels[:, 1] < height)
        )
        depth = np.full(len(depth_pixels), np.nan)
        depth[inside] = (
            self.depth_image[depth_pixels[inside, 1], depth_pixels[inside, 0]]
            / self.millimetres_per_metre
        )
        depth[depth <= 0.0] = np.nan

        principal_point = self.camera_model.principal_point
        return np.column_stack(
            [
                (depth_pixels[:, 0] - float(principal_point.x))
                * depth
                / self.camera_model.focal_length_x,
                (depth_pixels[:, 1] - float(principal_point.y))
                * depth
                / self.camera_model.focal_length_y,
                depth,
            ]
        )

    def world_points(
        self, color_pixels: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float64]:
        """
        :param color_pixels: The (x, y) of color image pixels, one pixel per row.
        :return: The (x, y, z) of the world point the camera saw at each of those
            pixels, one point per row, ``NaN`` where it measured no depth or the pixel
            lies outside the image.
        """
        camera_points = self.camera_points(color_pixels)
        homogeneous = np.column_stack([camera_points, np.ones(len(camera_points))])
        return (homogeneous @ self.world_T_camera.to_np().T)[:, :3]

    def world_point(self, color_pixel: Point2D) -> Point3 | None:
        """
        :param color_pixel: A color image pixel.
        :return: The world point the camera saw at the pixel, or ``None`` if it
            measured no depth there.
        """
        coordinates = self.world_points(np.array([[color_pixel.x, color_pixel.y]]))[0]
        if np.isnan(coordinates).any():
            return None
        return Point3(*coordinates, reference_frame=self.world_T_camera.reference_frame)


@dataclass
class DepthFrame:
    """
    The depth measurements point clouds are built from, with matching color.
    """

    depth_image: npt.NDArray[np.uint16]
    """
    The depth image.
    """

    rgb_image: npt.NDArray[np.uint8]
    """
    The color image in RGB order, scaled to the depth image resolution.
    """

    color_to_depth_ratio: ColorToDepthRatio
    """
    Size of the depth image relative to the color image.
    """

    camera_model: PinholeCameraModel
    """
    Projection model at the depth image resolution.
    """

    @classproperty
    def mask_foreground(cls) -> int:
        """
        Value of the pixels a point cloud is built from in a depth mask.
        """
        return 255

    @classmethod
    def from_cas(cls, cas: CAS) -> DepthFrame:
        """
        :param cas: A CAS holding a color image, a depth image and a camera observation.
        :return: The CAS's depth image with its color image and camera model scaled to
            the depth image's resolution.
        :raises CameraObservationMissing: If the CAS has no camera observation.
        """
        color_to_depth_ratio = cas.color_to_depth_ratio
        scaled_color = get_scaled_color_image_for_depth_image(
            cas, cas.get(CASViews.COLOR_IMAGE)
        )
        return cls(
            depth_image=cas.get(CASViews.DEPTH_IMAGE),
            rgb_image=cv2.cvtColor(scaled_color, cv2.COLOR_BGR2RGB),
            color_to_depth_ratio=color_to_depth_ratio,
            camera_model=cas.require_camera_observation().effective_camera_model.scaled(
                scale_x=color_to_depth_ratio.horizontal,
                scale_y=color_to_depth_ratio.vertical,
            ),
        )

    def box_cloud(
        self, *, box: Rect, depth_truncate: float
    ) -> open3d.geometry.PointCloud:
        """
        Build the point cloud of the depth measurements inside a box.

        :param box: The box in color image pixels.
        :param depth_truncate: Distance in meters beyond which measurements are left
            out.
        :return: The colored point cloud of the depth measurements inside the box.
        """
        left, top, right, bottom = self.color_to_depth_ratio.depth_region(
            box
        ).get_corner_points()
        mask = np.zeros_like(self.depth_image, dtype=np.uint8)
        mask[top:bottom, left:right] = self.mask_foreground
        return self.masked_cloud(mask=mask, depth_truncate=depth_truncate)

    def disc_cloud(
        self, *, x: float, y: float, radius: int
    ) -> open3d.geometry.PointCloud:
        """
        Build the point cloud of the depth measurements inside a disc.

        :param x: Horizontal disc center in color image pixels.
        :param y: Vertical disc center in color image pixels.
        :param radius: Disc radius in depth image pixels.
        :return: The colored point cloud of the depth measurements inside the disc.
        """
        center = self.color_to_depth_ratio.depth_pixel(x, y)
        mask = np.zeros_like(self.depth_image, dtype=np.uint8)
        cv2.circle(
            mask,
            (center.x, center.y),
            radius,
            self.mask_foreground,
            thickness=cv2.FILLED,
        )
        return self.masked_cloud(mask=mask)

    def masked_cloud(
        self, *, mask: npt.NDArray[np.uint8], **cloud_settings: float
    ) -> open3d.geometry.PointCloud:
        """
        Build the point cloud of the depth measurements a mask marks.

        :param mask: Mask on the depth image whose pixels with value
            :attr:`mask_foreground` the cloud is built from.
        :param cloud_settings: Further settings of
            :func:`~robokudo.utils.o3d_helper.get_cloud_from_rgb_depth_and_mask`.
        :return: The colored point cloud of the depth measurements inside the mask.
        """
        return get_cloud_from_rgb_depth_and_mask(
            rgb_image=self.rgb_image,
            depth_image=self.depth_image,
            mask=mask,
            camera_model=self.camera_model,
            mask_true_value=self.mask_foreground,
            **cloud_settings,
        )
