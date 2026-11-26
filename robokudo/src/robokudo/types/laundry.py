"""
What perception finds out about laundry in a basket so that a robot can pick it up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np
import numpy.typing as npt
from krrood_spatial_math.spatial_types import Pose

from robokudo.types.annotation import Classification
from robokudo.types.core import Annotation
from robokudo.types.cv import Point2D
from robokudo.types.scene import ObjectHypothesis

# %% the objects


class DetectedClassName(StrEnum):
    """
    Class names that a detector classifies objects by; members are added by subclasses.
    """

    @classmethod
    def classifies(cls, hypothesis: ObjectHypothesis) -> bool:
        """
        :param hypothesis: A detected object.
        :return: Whether any classification of the object names one of these classes.
        """
        return any(
            annotation.classname in cls
            for annotation in hypothesis.annotations
            if isinstance(annotation, Classification)
        )


class LaundryBasketClassName(DetectedClassName):
    """
    The class names a laundry basket is detected by.
    """

    LAUNDRY_BASKET = "laundry basket"
    """
    A basket named for the laundry it holds.
    """

    BASKET = "basket"
    """
    A basket of any kind, taken to hold laundry.
    """

    BOX = "box"
    """
    A box, taken to hold laundry.
    """


class LaundryClassName(DetectedClassName):
    """
    The class names laundry is detected by.
    """

    CLOTHES = "clothes"
    """
    Any garments.
    """

    PILE_OF_CLOTHES = "pile of clothes"
    """
    Garments lying on top of each other.
    """

    HEAP_OF_CLOTHES = "heap of clothes"
    """
    Garments lying on top of each other in disorder, the name the detector finds a
    basket's laundry by most confidently.
    """

    CLOTHES_TO_WASH = "laundry"
    """
    Garments gathered for washing.
    """

    T_SHIRT = "t-shirt"
    """
    A single T-shirt.
    """

    SHIRT = "shirt"
    """
    A single shirt, or a T-shirt taken for one.
    """

    GARMENT = "garment"
    """
    A single piece of clothing of any kind.
    """


# %% the basket and the laundry in it


@dataclass(kw_only=True)
class BasketFootprint:
    """
    The outline a laundry basket covers on the floor, seen from above in the world
    frame.
    """

    corners: npt.NDArray[np.float64]
    """
    The (x, y) corners of the outline's convex hull in the world frame,
    counter-clockwise, one per row.
    """

    @classmethod
    def from_points(cls, *, points: npt.NDArray[np.float64]) -> BasketFootprint:
        """
        :param points: The (x, y, z) world points measured on the basket, one per row.
        :return: The footprint enclosing the points seen from above.
        """
        hull = cv2.convexHull(points[:, :2].astype(np.float32), clockwise=False)
        return cls(corners=hull[:, 0, :].astype(np.float64))

    def distances_inside(
        self, *, points: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        :param points: World points with (x, y) in their first two columns, one per row.
        :return: The horizontal distance from each point to the footprint's outline,
            positive inside and negative outside.
        """
        edges = np.roll(self.corners, -1, axis=0) - self.corners
        inward_normals = np.column_stack([-edges[:, 1], edges[:, 0]])
        inward_normals /= np.linalg.norm(inward_normals, axis=1, keepdims=True)
        offsets = points[:, None, :2] - self.corners[None, :, :]
        return (offsets * inward_normals[None, :, :]).sum(axis=2).min(axis=1)


@dataclass(kw_only=True)
class LaundrySurfaceShape:
    """
    How the laundry's surface curves around a point, as a quadratic height surface
    fitted in the world frame.
    """

    across_curvature: float
    """
    The curvature in the direction the surface curves down most strongly, in 1/m,
    negative where it curves down.
    """

    along_curvature: float
    """
    The curvature in the direction perpendicular to :attr:`across_direction`, in 1/m.
    """

    across_direction: npt.NDArray[np.float64]
    """
    The flat world direction of :attr:`across_curvature`, of unit length.
    """

    def has_ridge(self, *, dominance: float) -> bool:
        """
        :param dominance: How many times as strongly the surface has to curve down
            across the ridge as along it.
        :return: Whether the surface forms a ridge along the direction perpendicular
            to :attr:`across_direction`.
        """
        return self.across_curvature < 0.0 and abs(
            self.across_curvature
        ) >= dominance * abs(self.along_curvature)


# %% picking up


@dataclass(kw_only=True)
class LaundryGraspAnnotation(Annotation):
    """
    Where to take hold of the laundry in a basket, at the highest point of the laundry.
    """

    pixel: Point2D
    """
    The color image pixel of the highest point of the laundry.
    """

    height_above_floor: float
    """
    How high the highest point of the laundry lies above the floor, in metres.
    """

    grasp_pose: Pose
    """
    The grasp in the world frame, approached along its x-axis and closed along its
    y-axis.
    """

    @property
    def position(self) -> npt.NDArray[np.float64]:
        """
        The (x, y, z) of the grasp in the world frame.
        """
        return self.grasp_pose.to_np()[:3, 3]


@dataclass(kw_only=True)
class SteadyLaundryGraspAnnotation(Annotation):
    """
    The laundry grasp held steady across the frames of a query stream.
    """

    grasp: LaundryGraspAnnotation
    """
    The grasp, found in this frame or an earlier one of the stream.
    """


# %% holding the grasp steady


@dataclass(kw_only=True)
class SteadyLaundryGrasp:
    """
    The laundry grasp held across the frames of a query stream. It follows a grasp found
    elsewhere only once the grasp stays there for :attr:`confirmation_frames` frames,
    and is forgotten once no grasp is found for as many frames.
    """

    stay_radius: float = 0.03
    """
    How far, in metres, a grasp may lie from the held one to count as the same spot.
    """

    confirmation_frames: int = 3
    """
    For how many frames in a row a grasp has to stay elsewhere, or be missing, before
    the held grasp follows.
    """

    history_length: int = 5
    """
    How many of the latest grasps at the held spot are kept to choose from.
    """

    held: list[LaundryGraspAnnotation] = field(default_factory=list)
    """
    The latest grasps at the held spot, oldest first.
    """

    candidates: list[LaundryGraspAnnotation] = field(default_factory=list)
    """
    The grasps found in a row at one spot other than the held one, oldest first.
    """

    frames_without_grasp: int = 0
    """
    For how many frames in a row no grasp was found.
    """

    @property
    def steady_grasp(self) -> LaundryGraspAnnotation | None:
        """
        The most central of the latest grasps at the held spot, ``None`` if no grasp is
        held.
        """
        if len(self.held) == 0:
            return None
        return self.most_central(grasps=self.held)

    def follow(self, *, grasp: LaundryGraspAnnotation | None) -> None:
        """
        Take a frame's grasp into account.

        :param grasp: The grasp found in the frame, ``None`` if none was found.
        """
        if grasp is None:
            self.frames_without_grasp += 1
            if self.frames_without_grasp >= self.confirmation_frames:
                self.held = []
                self.candidates = []
            return
        self.frames_without_grasp = 0
        if len(self.held) == 0 or self.is_near(grasp=grasp, grasps=self.held):
            self.held = [*self.held, grasp][-self.history_length :]
            self.candidates = []
            return
        if len(self.candidates) > 0 and not self.is_near(
            grasp=grasp, grasps=self.candidates
        ):
            self.candidates = []
        self.candidates = [*self.candidates, grasp]
        if len(self.candidates) >= self.confirmation_frames:
            self.held = self.candidates[-self.history_length :]
            self.candidates = []

    def is_near(
        self, *, grasp: LaundryGraspAnnotation, grasps: list[LaundryGraspAnnotation]
    ) -> bool:
        """
        :param grasp: A grasp found in a frame.
        :param grasps: Grasps found at one spot.
        :return: Whether the grasp lies within :attr:`stay_radius` of the most central
            of the grasps.
        """
        center = self.most_central(grasps=grasps).position
        return bool(np.linalg.norm(grasp.position - center) <= self.stay_radius)

    @staticmethod
    def most_central(*, grasps: list[LaundryGraspAnnotation]) -> LaundryGraspAnnotation:
        """
        :param grasps: At least one grasp.
        :return: The grasp lying closest to the median position of the grasps.
        """
        positions = np.array([grasp.position for grasp in grasps])
        offsets = np.linalg.norm(positions - np.median(positions, axis=0), axis=1)
        return grasps[int(np.argmin(offsets))]
