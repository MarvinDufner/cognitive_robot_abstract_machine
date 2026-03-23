"""Computer vision types for Robokudo.

This module provides types for computer vision operations including:

* 2D and 3D point representations
* Rectangle and region of interest definitions
* 3D bounding box specifications

The types support integration with:
* OpenCV for image processing
* Open3D for point cloud handling
* Transform system for poses
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING
import open3d as o3d

from robokudo.types.core import Type
from robokudo.types.tf import Pose

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass
class Point2D(Type):
    """2D point representation.

    Represents a point in 2D image coordinates.
    """

    x: int = 0
    """
    X coordinate
    """

    y: int = 0
    """
    Y coordinate
    """


@dataclass
class Points3D(Type):
    """3D point cloud container.

    Wraps an Open3D point cloud for 3D point operations.
    """

    points: o3d.geometry.PointCloud = None
    """
    The actual Open3D point cloud object
    """


@dataclass
class Rect(Type):
    """2D rectangle representation.

    Defines a rectangle by its top-left corner position and dimensions.
    """

    pos: Point2D = field(default_factory=Point2D)
    """
    Top-left corner position
    """

    width: int = 0
    """
    Rectangle width in pixels
    """

    height: int = 0
    """
    Rectangle height in pixels
    """


@dataclass
class ImageROI(Type):
    """Image region of interest.

    Defines a region of interest in an image using:
    * Binary mask for arbitrary shapes
    * Rectangle for bounding region
    """

    mask: npt.NDArray = None
    """
    Binary opencv mask image
    """

    roi: Rect = field(default_factory=Rect)
    """
    Rectangular region of interest
    """


@dataclass
class BoundingBox3D(Type):
    """3D oriented bounding box.

    Represents a 3D box with:
    * Dimensions along each axis
    * 6-DOF pose defining orientation and position
    """

    x_length: float = 0.0
    """
    Box length along x-axis
    """

    y_length: float = 0.0
    """
    Box length along y-axis
    """

    z_length: float = 0.0
    """
    Box length along z-axis
    """

    pose: Pose = field(default_factory=Pose)
    """
    Box pose in 3D space
    """
