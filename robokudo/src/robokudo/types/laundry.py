"""
What perception finds out about laundry in a basket so that a robot can pick it up.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

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
    Any garments, the name the detector finds a heap of them by.
    """


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
