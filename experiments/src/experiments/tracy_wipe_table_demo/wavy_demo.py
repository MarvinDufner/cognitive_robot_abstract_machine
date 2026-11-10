"""
Tracy wipes its bench with ridges laid across the wiped patch, so the tool has to follow
a surface nothing told it about.

The plan is the one of :mod:`experiments.tracy_wipe_table_demo.demo`: the descent feels
for the first crest it meets and the press holds the force from there on, so how well
contact is kept over the crests is what this run measures.

The execution type is the first command line argument and the wiping technique the
second, both defaulting to what :func:`main` takes::

    python3 wavy_demo.py PHYSICALLY_SIMULATED WIPE

.. warning:: Only a physically simulated run resolves the ridges. A kinematically
    simulated one models a flat surface at the height of the bench top and feels
    nothing of them, and a real run would wipe the bare bench while the plan believes
    the ridges are there.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from coraplex.datastructures.enums import WipingTechnique
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor, PlanExecutor
from experiments.wipe_table_demo.demo import EXECUTORS
from experiments.tracy_wipe_table_demo.demo import TracyWipeTableDemonstration
from experiments.wipe_table_demo.demo import RidgedSurface
from semantic_digital_twin.robots.tracy import Tracy

# %% the demonstration


@dataclass
class TracyWipeWavyBenchDemonstration(RidgedSurface, TracyWipeTableDemonstration):
    """
    Tracy wipes a bench with ridges across it.
    """

    ridged_span: float = 0.34
    """
    The wiped patch's length along the sweep, in m.
    """


def main(
    executor_type: type[PlanExecutor] = PhysicallySimulatedPlanExecutor,
    technique: WipingTechnique = TracyWipeWavyBenchDemonstration.technique,
) -> None:
    """
    Run the demonstration.

    :param executor_type: The executor running the plan, which decides whether it drives
        the real robot or simulates it.
    :param technique: How the tool covers the patch.
    """
    TracyWipeWavyBenchDemonstration(
        used_robot=Tracy, executor_type=executor_type, technique=technique
    ).run()


if __name__ == "__main__":
    # Only what the command line names, so each default lives once, in main.
    main(
        *(EXECUTORS[name] for name in sys.argv[1:2]),
        *(WipingTechnique[name] for name in sys.argv[2:3]),
    )
