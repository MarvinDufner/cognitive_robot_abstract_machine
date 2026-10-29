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

import numpy as np

from coraplex.datastructures.enums import ExecutionType, WipingTechnique
from experiments.tracy_wipe_table_demo.demo import TracyWipeTableDemonstration
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% the demonstration


@dataclass
class TracyWipeWavyBenchDemonstration(TracyWipeTableDemonstration):
    """
    Tracy wipes a bench with ridges across it.
    """

    wave_amplitude: float = 0.02
    """
    How far the crests stand above the bench top, in m.
    """

    wave_length: float = 0.12
    """
    Distance between two crests, in m, along the sweep.
    """

    wave_radius: float = 0.06
    """
    Radius of the ridges, in m: the curvature the tool is thrown off by at a crest.
    """

    ridge_length: float = 0.6
    """
    Length of each ridge across the bench, in m: wider than the wiped patch.
    """

    def populate_scene(self, world: World) -> None:
        """
        Bolt the sponge into the gripper, then lay the ridges across the wiped patch.
        """
        super().populate_scene(world)
        bench_top = self.bench_height(world)
        centre = self.wiped_pose(world).to_np()[:3, 3]
        crests = int(0.5 * self.patch_length / self.wave_length) + 2
        with world.modify_world():
            for index in range(-crests, crests + 1):
                world.add_connection(
                    FixedConnection(
                        parent=world.root,
                        child=self.ridge(f"ridge_{index + crests}"),
                        parent_T_connection_expression=(
                            HomogeneousTransformationMatrix.from_xyz_rpy(
                                x=centre[0] + index * self.wave_length,
                                y=centre[1],
                                # Sunk into the bench, so only the crest stands out of
                                # it and the tool rides a wave rather than a step.
                                z=bench_top + self.wave_amplitude - self.wave_radius,
                                roll=np.pi / 2.0,
                            )
                        ),
                    )
                )

    def ridge(self, name: str) -> Body:
        """
        :param name: Name the ridge is added under.
        :return: One ridge, lying across the bench.
        """
        return Body(
            name=PrefixedName(name),
            collision=ShapeCollection([self.ridge_shape()]),
            visual=ShapeCollection([self.ridge_shape()]),
        )

    def ridge_shape(self) -> Cylinder:
        """
        :return: The shape of one ridge, as its own instance, since a body's collision
            and its visual each own theirs.
        """
        return Cylinder(width=2.0 * self.wave_radius, height=self.ridge_length)


def main(
    execution_type: ExecutionType = ExecutionType.PHYSICALLY_SIMULATED,
    technique: WipingTechnique = TracyWipeWavyBenchDemonstration.technique,
) -> None:
    """
    Run the demonstration.

    :param execution_type: Whether to drive the real robot or simulate it.
    :param technique: How the tool covers the patch.
    """
    TracyWipeWavyBenchDemonstration(
        used_robot=Tracy, execution_type=execution_type, technique=technique
    ).run()


if __name__ == "__main__":
    # Only what the command line names, so each default lives once, in main.
    main(
        *(ExecutionType[name] for name in sys.argv[1:2]),
        *(WipingTechnique[name] for name in sys.argv[2:3]),
    )
