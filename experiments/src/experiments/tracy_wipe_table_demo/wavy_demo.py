"""
Tracy wipes its bench with ridges laid across the wiped patch, so the tool has to follow
a surface nothing told it about.

The plan is the one of :mod:`experiments.tracy_wipe_table_demo.demo`, so how well
contact is kept over the crests is what this run measures.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.datastructures.enums import WipingTechnique
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor, PlanExecutor
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

    :param executor_type: The executor running the plan.
    :param technique: How the tool covers the patch.
    """
    TracyWipeWavyBenchDemonstration(
        used_robot=Tracy, executor_type=executor_type, technique=technique
    ).run()


if __name__ == "__main__":
    main()
