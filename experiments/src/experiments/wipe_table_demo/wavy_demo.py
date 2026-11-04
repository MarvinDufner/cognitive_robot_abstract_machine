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

    wipe_speed: float = 0.025
    """
    Slower than the flat table is wiped at, in m/s.

    What the press has to absorb at a crest is the speed the tool is driven into the
    rising surface at, and halving it is worth more than any gain of the press's own:
    the HSRB pressing 15 N over these crests peaks at 42 N and holds contact 82 % of the
    time at 4.5 cm/s, and at 36 N and 89 % here.
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
