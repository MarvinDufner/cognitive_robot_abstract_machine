"""
A synthetic camera frame of a table, seen from straight above.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point2
from robokudo.cas import CAS, CASViews
from robokudo.types.camera import CameraObservation
from robokudo.types.cv import ColorToDepthRatio
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world_description.world_entity import Body

# %% the scene


@dataclass
class TableFromAboveScene(ABC):
    """
    A camera one metre above a table looks straight down at what lies on it.

    The world's x-axis runs to the right of the image and its z-axis up from the table.
    """

    image_size: int = 200
    """
    Width and height of the color and depth images, in pixels.
    """

    table_depth: int = 1000
    """
    Distance from the camera to the table, in millimetres.
    """

    table_gray_level: int = 100
    """
    Brightness of the empty table.
    """

    world_frame: Body = field(default_factory=lambda: Body(name=PrefixedName("world")))
    """
    The frame the camera's pose is given in.
    """

    @property
    def camera_model(self) -> PinholeCameraModel:
        """
        A camera with a focal length of 100 pixels, centered on the image.
        """
        return PinholeCameraModel(
            image_resolution=CameraResolution(
                width=self.image_size, height=self.image_size
            ),
            focal_length_x=100.0,
            focal_length_y=100.0,
            principal_point=Point2(x=self.image_size / 2, y=self.image_size / 2),
        )

    def table_image(self) -> npt.NDArray[np.uint8]:
        """
        :return: The empty table.
        """
        return np.full(
            (self.image_size, self.image_size, 3),
            self.table_gray_level,
            dtype=np.uint8,
        )

    def table_depth_image(self) -> npt.NDArray[np.uint16]:
        """
        :return: The distance to the empty table.
        """
        return np.full(
            (self.image_size, self.image_size), self.table_depth, dtype=np.uint16
        )

    @abstractmethod
    def color_image(self) -> npt.NDArray[np.uint8]:
        """
        :return: The table with what lies on it.
        """

    @abstractmethod
    def depth_image(self) -> npt.NDArray[np.uint16]:
        """
        :return: The distance to the table and to what lies on it, in millimetres.
        """

    def camera_observation(self) -> CameraObservation:
        """
        :return: The camera one metre above the table's surface, looking down.
        """
        camera_body = Body(name=PrefixedName("camera"))
        return CameraObservation(
            camera=Camera(
                name=PrefixedName("camera"),
                root=camera_body,
                camera_model=self.camera_model,
            ),
            effective_camera_model=self.camera_model,
            world_T_camera=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=self.table_depth / 1000,
                roll=math.pi,
                reference_frame=self.world_frame,
                child_frame=camera_body,
            ),
            timestamp_nanoseconds=0,
        )

    def cas(self) -> CAS:
        """
        :return: A CAS holding the camera frame.
        """
        cas = CAS()
        cas.set(CASViews.COLOR_IMAGE, self.color_image())
        cas.set(CASViews.DEPTH_IMAGE, self.depth_image())
        cas.set(CASViews.COLOR_TO_DEPTH_RATIO, ColorToDepthRatio())
        cas.camera_observation = self.camera_observation()
        return cas
