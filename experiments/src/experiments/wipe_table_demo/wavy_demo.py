"""
The HSRB wipes a kitchen table with ridges laid across it, so the tool has to follow a
surface nothing told it about.

The plan is the one of :mod:`experiments.wipe_table_demo.demo`, so how well contact is
kept over the crests is what this run measures.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.datastructures.enums import WipingTechnique
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor, PlanExecutor
from experiments.wipe_table_demo.demo import RidgedSurface, WipeTableDemonstration
from semantic_digital_twin.robots.hsrb import HSRB

# %% the demonstration


@dataclass
class WipeWavyTableDemonstration(RidgedSurface, WipeTableDemonstration):
    """
    The HSRB wipes a dining table with ridges across it.

    The plan is the flat demonstration's, so how well contact is kept over the crests is
    what this run measures.
    """

    ros_node_name: str = "wipe_wavy_table_demo_node"
    """
    Name of the node this demonstration talks to ROS under.
    """

    wipe_speed: float = 0.025
    """
    Slower than the flat table is wiped at, in m/s.

    What the press has to absorb at a crest is the speed the tool is driven into the
    rising surface at: the HSRB pressing 15 N over these crests peaks at 42 N and holds
    contact 82 % of the time at 4.5 cm/s, and at 36 N and 89 % here.
    """

    ridged_span: float = 0.95
    """
    How far the crests reach along the lanes, in m.

    The lanes run along the table's long side, and this is as far as the crests cover
    them while every crest still lands on the table rather than off its ends.
    """

    ridge_length: float = 0.15
    """
    Length of each ridge, in m.

    The lanes of a patch a parked HSRB reaches are only a hand's width apart across, so
    a ridge this long carries all of them and still ends short of the table's edge.
    """


def main(
    executor_type: type[PlanExecutor] = PhysicallySimulatedPlanExecutor,
    technique: WipingTechnique = WipeWavyTableDemonstration.technique,
) -> None:
    """
    Run the demonstration.

    :param executor_type: The executor running the plan.
    :param technique: How the tool covers the table.
    """
    WipeWavyTableDemonstration(
        used_robot=HSRB, executor_type=executor_type, technique=technique
    ).run()


if __name__ == "__main__":
    main()
