"""
The HSRB wipes a kitchen table with ridges laid across it, so the tool has to follow a
surface nothing told it about.

The plan is the one of :mod:`experiments.wipe_table_demo.demo`: the descent feels for
the first crest it meets and the press holds the force from there on, so how well
contact is kept over the crests is what this run measures.

The execution type is the first command line argument and the wiping technique the
second, both defaulting to what :func:`main` takes::

    python3 wavy_demo.py PHYSICALLY_SIMULATED SPREAD

.. warning:: Only a physically simulated run resolves the ridges. A kinematically
    simulated one models a flat surface at the height of the table top and feels
    nothing of them, and a real run would wipe the bare table while the plan believes
    the ridges are there.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from typing_extensions import ClassVar

from coraplex.datastructures.enums import ExecutionType, WipingTechnique
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

    ros_node_name: ClassVar[str] = "wipe_wavy_table_demo_node"

    ridged_span: float = 0.72
    """
    How far the ridges reach along the sweep, in m: the table's depth, which the crests
    then cover without standing off its edges, where the base drives.
    """

    ridge_length: float = 1.15
    """
    Length of each ridge across the table, in m: wider than the lanes the wipe lays
    down, and still short of the table's edges.
    """


def main(
    execution_type: ExecutionType = ExecutionType.PHYSICALLY_SIMULATED,
    technique: WipingTechnique = WipeWavyTableDemonstration.technique,
) -> None:
    """
    Run the demonstration.

    :param execution_type: Whether to drive the real robot or simulate it.
    :param technique: How the tool covers the table.
    """
    WipeWavyTableDemonstration(
        used_robot=HSRB, execution_type=execution_type, technique=technique
    ).run()


if __name__ == "__main__":
    # Only what the command line names, so the default lives once, in main.
    main(
        *(ExecutionType[name] for name in sys.argv[1:2]),
        *(WipingTechnique[name] for name in sys.argv[2:3]),
    )
