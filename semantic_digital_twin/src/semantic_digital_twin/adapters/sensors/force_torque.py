"""
Where the reading of a force/torque sensor comes from while no real sensor is attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.wrench import Wrench
from semantic_digital_twin.input_synchronization import InputSynchronizer
from semantic_digital_twin.robots.input_source import SimulatedWrenchSource
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.world_description.world_entity import Body

# %% what a tool is pressed with


@dataclass
class PhysicallySimulatedContactWrench(SimulatedWrenchSource, InputSynchronizer):
    """
    Writes what a physics simulation presses a tool with into the force/torque sensor
    measuring for it, in place of a real sensor's topic.

    The simulation resolves the contact, so the reading holds whatever the tool touches
    rather than one modelled surface.
    """

    tool: Body = field(kw_only=True)
    """
    The body that touches what it is pressed against.
    """

    simulation: MujocoSim = field(kw_only=True)
    """
    The running simulation the contacts are read from.
    """

    def apply(self) -> bool:
        """
        Write the contact wrench, and report that nothing has to be announced: a wrench
        leaves the kinematic state untouched.
        """
        sensor = ForceTorqueSensor.for_tip(self.world, self.tool)
        world_T_sensor = self.world.compute_forward_kinematics_np(
            self.world.root, sensor.root
        )
        wrench = self.simulation.simulator.get_contact_wrench(
            body_names=[self.tool.name.name], about_position=world_T_sensor[:3, 3]
        ).result
        sensor.write_wrench(
            Wrench(force=wrench[:3], torque=wrench[3:]).rotated_into(
                world_T_sensor[:3, :3].T
            )
        )
        return False
