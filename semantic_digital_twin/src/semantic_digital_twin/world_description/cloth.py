"""
Cloth a physical simulation deforms, and the flat patterns garments are cut from.

A physics engine reads a :class:`ClothSheet` off the world as a simulator property; the
world's kinematic structure holds no cloth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy
import numpy.typing as npt
import trimesh

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.mixin import SimulatorAdditionalProperty
from semantic_digital_twin.world_description.geometry import Color

# %% cloth sheets


@dataclass
class ClothElasticity:
    """
    How stiffly a cloth's material resists bending, as the elastic constants of a thin
    shell.
    """

    young_modulus: float = 3e5
    """
    Young's modulus of the material, in pascals.
    """

    poisson_ratio: float = 0.0
    """
    Poisson's ratio of the material.
    """


@dataclass
class ClothSheet(SimulatorAdditionalProperty):
    """
    A piece of cloth a physical simulation deforms: a sheet of triangles whose edges keep
    their length, which bends elastically and collides as a shell of its thickness.
    """

    name: str
    """
    The name the simulation knows the cloth by.
    """

    surface: trimesh.Trimesh
    """
    The cloth's triangles as they lie when the simulation starts, with their vertices in
    the world frame.
    """

    mass: float
    """
    Mass of the whole cloth, in kilograms.
    """

    thickness: float = 8e-3
    """
    Thickness of the cloth, in metres: it collides as a shell this thick, and its bending
    stiffness grows with it.
    """

    elasticity: ClothElasticity = field(default_factory=ClothElasticity)
    """
    How stiffly the cloth resists bending.
    """

    color: Color = field(default_factory=Color)
    """
    The color the cloth is drawn in.
    """


# %% garment patterns


@dataclass
class TShirtPattern:
    """
    The flat outline a T-shirt lies in when spread out: a rectangular body with a
    rectangular sleeve on either side of its top and a notch for the neck.

    The outline follows a square grid, so every dimension is rounded to
    :attr:`vertex_spacing`.
    """

    body_width: float = 0.5
    """
    Width of the body from side seam to side seam, in metres.
    """

    body_length: float = 0.7
    """
    Length of the body from the hem to the shoulder line, in metres.
    """

    sleeve_length: float = 0.2
    """
    Length of a sleeve from the side seam to its cuff, in metres.
    """

    sleeve_width: float = 0.2
    """
    Width of a sleeve, from the shoulder line down to the armpit, in metres.
    """

    neck_width: float = 0.2
    """
    Width of the neck opening, in metres.
    """

    neck_depth: float = 0.05
    """
    How far the neck opening reaches down from the shoulder line, in metres.
    """

    vertex_spacing: float = 0.05
    """
    Distance between neighbouring vertices of the grid the outline is cut from, in
    metres.
    """

    def create_surface(
        self, world_T_pattern: HomogeneousTransformationMatrix
    ) -> trimesh.Trimesh:
        """
        :param world_T_pattern: Where the pattern lies: its origin is the centre of the
            body, its x-axis points along the sleeves and its y-axis from the hem to the
            neck.
        :return: The pattern as a sheet of triangles lying flat in its x-y plane, with
            its vertices in the world frame.
        """
        half_span = self.body_width / 2 + self.sleeve_length
        grid_x = numpy.linspace(
            -half_span, half_span, round(2 * half_span / self.vertex_spacing) + 1
        )
        grid_y = numpy.linspace(
            -self.body_length / 2,
            self.body_length / 2,
            round(self.body_length / self.vertex_spacing) + 1,
        )
        column_count = len(grid_x) - 1
        rows, columns = numpy.divmod(
            numpy.arange(column_count * (len(grid_y) - 1)), column_count
        )
        covered = self._covers(
            (grid_x[columns] + grid_x[columns + 1]) / 2,
            (grid_y[rows] + grid_y[rows + 1]) / 2,
        )
        lower_left = rows[covered] * len(grid_x) + columns[covered]
        lower_right = lower_left + 1
        upper_left = lower_left + len(grid_x)
        upper_right = upper_left + 1
        x, y = numpy.meshgrid(grid_x, grid_y)
        surface = trimesh.Trimesh(
            vertices=numpy.column_stack([x.ravel(), y.ravel(), numpy.zeros(x.size)]),
            faces=numpy.concatenate(
                [
                    numpy.column_stack([lower_left, lower_right, upper_right]),
                    numpy.column_stack([lower_left, upper_right, upper_left]),
                ]
            ),
        )
        surface.remove_unreferenced_vertices()
        return surface.apply_transform(world_T_pattern.to_np())

    def _covers(
        self, x: npt.NDArray[numpy.float64], y: npt.NDArray[numpy.float64]
    ) -> npt.NDArray[numpy.bool_]:
        """
        :param x: The x-coordinates of points in the pattern's frame.
        :param y: The y-coordinates of the same points.
        :return: Which of the points lie within the outline.
        """
        distance_from_middle = numpy.abs(x)
        below_shoulder_line = self.body_length / 2 - y
        in_neck = (distance_from_middle < self.neck_width / 2) & (
            below_shoulder_line < self.neck_depth
        )
        in_body = (distance_from_middle < self.body_width / 2) & ~in_neck
        in_sleeve = (
            (distance_from_middle >= self.body_width / 2)
            & (distance_from_middle < self.body_width / 2 + self.sleeve_length)
            & (below_shoulder_line < self.sleeve_width)
        )
        return in_body | in_sleeve
