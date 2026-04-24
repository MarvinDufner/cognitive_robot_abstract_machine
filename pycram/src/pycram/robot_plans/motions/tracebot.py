from dataclasses import dataclass
from typing import Optional

import numpy as np

from giskardpy.motion_statechart.goals.tracebot import InsertCylinder
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.world_entity import Body

from pycram.robot_plans.motions.base import BaseMotion


@dataclass
class InsertCanisterMotion(BaseMotion):
    """Motion that inserts a cylinder-shaped object into a hole using InsertCylinder."""

    cylinder: Body
    """The cylinder body to insert."""

    hole_point: Point3
    """The target hole position in world space."""

    cylinder_height: Optional[float] = None
    up: Optional[Vector3] = None
    pre_grasp_height: float = 0.1
    tilt: float = np.pi / 10
    get_straight_after: float = 0.02

    def perform(self):
        pass

    @property
    def _motion_chart(self) -> InsertCylinder:
        return InsertCylinder(
            cylinder_name=self.cylinder,
            hole_point=self.hole_point,
            cylinder_height=self.cylinder_height,
            up=self.up,
            pre_grasp_height=self.pre_grasp_height,
            tilt=self.tilt,
            get_straight_after=self.get_straight_after,
        )
