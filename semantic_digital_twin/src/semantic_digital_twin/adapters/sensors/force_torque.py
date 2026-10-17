"""
Where the reading of a force/torque sensor comes from while no real sensor is attached.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Tuple

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.input_synchronization import InputSynchronizer
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.world_description.world_entity import Body

# %% what a tool is pressed with


@dataclass
class ToolContactWrench(InputSynchronizer, ABC):
    """
    Writes what a tool is pressed with into the force/torque sensor measuring for it, in
    place of a real sensor's topic.
    """

    tool: Body = field(kw_only=True)
    """
    The body that touches what it is pressed against.
    """

    def apply(self) -> bool:
        """
        Write the contact wrench, and report that nothing has to be announced: a wrench
        leaves the kinematic state untouched.
        """
        sensor = ForceTorqueSensor.for_tip(self.world, self.tool)
        force, torque = self.contact_wrench(sensor)
        sensor.write_wrench(force, torque)
        return False

    @abstractmethod
    def contact_wrench(
        self, sensor: ForceTorqueSensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param sensor: The sensor the wrench is measured by.
        :return: The force and the torque about that sensor's own origin, both in its
            frame.
        """


@dataclass
class PhysicallySimulatedContactWrench(ToolContactWrench):
    """
    What a physics simulation presses the tool with, resolved by the simulation itself
    rather than modelled for one surface.
    """

    simulation: MujocoSim = field(kw_only=True)
    """
    The running simulation the contacts are read from.
    """

    def contact_wrench(
        self, sensor: ForceTorqueSensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        world_T_sensor = self.world.compute_forward_kinematics_np(
            self.world.root, sensor.root
        )
        wrench = self.simulation.simulator.get_contact_wrench(
            body_names=[self.tool.name.name], about_position=world_T_sensor[:3, 3]
        ).result
        sensor_R_world = world_T_sensor[:3, :3].T
        return sensor_R_world @ wrench[:3], sensor_R_world @ wrench[3:]
