"""
The garments the folding bench lays out: the cloth each is simulated as, lying spread
out, and where its landmarks lie on that cloth.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy
import trimesh
from typing_extensions import Generic

from krrood.patterns.path_enumeration import PathEnumeration
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.folding.garment import LandmarkType, TemplateType
from semantic_digital_twin.folding.t_shirt import TShirtTemplate
from semantic_digital_twin.folding.trousers import TrousersTemplate
from semantic_digital_twin.semantic_annotations.garments import TShirtLandmark
from semantic_digital_twin.world_description.cloth import (
    ClothElasticity,
    GarmentFile,
)

# %% spread garments


class GarmentResource(PathEnumeration):
    """
    The files the bench reads about its garments.
    """

    T_SHIRT_LANDMARK_VERTICES = Path(__file__).parent / "t_shirt_landmark_vertices.json"
    """
    The index of the vertex of :attr:`GarmentFile.T_SHIRT` at each
    :class:`TShirtLandmark`.
    """


@dataclass(kw_only=True)
class SpreadGarmentMesh:
    """
    The triangles of a garment lying spread out, in a frame of its own centred on it,
    with its relief above zero.
    """

    surface: trimesh.Trimesh
    """
    The garment's triangles.
    """

    landmark_vertices: list[int]
    """
    The index of the vertex at each landmark of the garment, in the order of the
    garment's landmarks.
    """


@dataclass(kw_only=True)
class BenchGarment(Generic[TemplateType], SubClassSafeGeneric, ABC):
    """
    A garment the bench lays out, and the fabric it is made of.
    """

    mass: float
    """
    The mass of the garment, in kilograms.
    """

    thickness: float = 1e-3
    """
    The thickness of the garment's fabric, in metres; thinner than the 3 mm RGBench
    states for its T-shirt, at which a folded sleeve springs back open in the
    simulation.
    """

    elasticity: ClothElasticity = field(
        default_factory=lambda: ClothElasticity(
            young_modulus=3 / 4 * (1 - 0.35) * 2.3e5, poisson_ratio=0.35
        )
    )
    """
    How stiffly the fabric resists bending: that of RGBench's T-shirt, as
    :class:`~experiments.t_shirt_demo.demo.TShirtSwingDemonstration` converts it.
    """

    @property
    @abstractmethod
    def spread_mesh(self) -> SpreadGarmentMesh:
        """
        The garment lying spread out.
        """

    @cached_property
    def template(self) -> TemplateType:
        """
        The garment lying spread out, given by where its landmarks lie in the frame of
        :attr:`spread_mesh`.
        """
        template_type = type(self).get_type_of_generic_parameter(TemplateType)
        landmarks = template_type.get_type_of_generic_parameter(LandmarkType)
        mesh = self.spread_mesh
        return template_type(
            landmark_positions=dict(
                zip(landmarks, mesh.surface.vertices[mesh.landmark_vertices, :2])
            )
        )


# %% the T-shirt


@dataclass(kw_only=True)
class BenchTShirt(BenchGarment[TShirtTemplate]):
    """
    RGBench's T-shirt, :attr:`GarmentFile.T_SHIRT`, stretched to a size of its own,
    lying with its hem along the y-axis of its frame.
    """

    collar_to_hem_length: float = 0.62
    """
    How long the T-shirt is from its collar to its hem, in metres.
    """

    sleeve_to_sleeve_width: float = 0.61
    """
    How wide the T-shirt is from the end of one sleeve to the end of the other, in
    metres.
    """

    mass: float = 0.13
    """
    The mass of RGBench's T-shirt, scaled with its area, in kilograms.
    """

    @cached_property
    def spread_mesh(self) -> SpreadGarmentMesh:
        surface = GarmentFile.T_SHIRT.create_surface(HomogeneousTransformationMatrix())
        length, width, _ = surface.extents
        surface.vertices = (surface.vertices - surface.bounds.mean(axis=0)) * [
            self.collar_to_hem_length / length,
            self.sleeve_to_sleeve_width / width,
            1.0,
        ]
        surface.vertices[:, 2] -= surface.vertices[:, 2].min()
        vertex_of_landmark = json.loads(
            GarmentResource.T_SHIRT_LANDMARK_VERTICES.read_text()
        )
        return SpreadGarmentMesh(
            surface=surface,
            landmark_vertices=[
                vertex_of_landmark[landmark] for landmark in TShirtLandmark
            ],
        )


# %% the trousers


@dataclass(kw_only=True)
class _TrousersPanel:
    """
    The front or the back of a pair of trousers, as rows of vertices across them.
    """

    positions: numpy.ndarray
    """
    Where each vertex lies, of shape (n, 2).
    """

    triangles: numpy.ndarray
    """
    The vertices of each triangle, counter-clockwise seen from above, of shape (m, 3).
    """

    on_seam: numpy.ndarray
    """
    Whether each vertex lies on a seam, where the front and the back are sewn together.
    """

    on_edge: numpy.ndarray
    """
    Whether each vertex lies on the panel's edge.
    """

    landmark_vertices: list[int]
    """
    The index of the vertex at each
    :class:`~semantic_digital_twin.semantic_annotations.garments.TrousersLandmark`, in
    their order.
    """


@dataclass(kw_only=True)
class BenchTrousers(BenchGarment[TrousersTemplate]):
    """
    A pair of trousers with straight seams, as in the pants model of Miller et al.,
    meshed as a front and a back sewn together along the outseams and the inseams and
    open at the waistband and the cuffs. They lie with their waistband along the x-axis
    of their frame, towards positive y, and their wearer's right towards negative x.
    """

    length: float = 0.98
    """
    How long the trousers are from the waistband to the cuffs, in metres.
    """

    waist_width: float = 0.42
    """
    How wide the waistband lies, in metres.
    """

    rise: float = 0.28
    """
    How far the crotch lies below the waistband, in metres.
    """

    cuff_width: float = 0.2
    """
    How wide each cuff lies, in metres.
    """

    cuff_gap: float = 0.14
    """
    How far apart the cuffs lie, in metres.
    """

    edge_length: float = 0.045
    """
    About how long the edges of the mesh's triangles are, in metres.
    """

    disorder: float = 0.3
    """
    How far each vertex inside a panel is moved at random off its row and column, as a
    share of :attr:`edge_length`, so that the edges do not line up along the folds.
    """

    seed: int = 0
    """
    Seeds where the vertices inside the panels are moved.
    """

    mass: float = 0.3
    """
    The mass of the trousers, in kilograms.
    """

    layer_gap: float = 3e-3
    """
    How far above the back the front is laid out, in metres.
    """

    @cached_property
    def spread_mesh(self) -> SpreadGarmentMesh:
        random_generator = numpy.random.default_rng(self.seed)
        panel = self._panel
        inside = ~panel.on_edge
        back = panel.positions.copy()
        front = panel.positions.copy()
        for positions in (back, front):
            positions[inside] += random_generator.uniform(
                -self.disorder * self.edge_length,
                self.disorder * self.edge_length,
                (int(inside.sum()), 2),
            )
        front_vertex = numpy.arange(len(front))
        front_vertex[~panel.on_seam] = len(back) + numpy.arange((~panel.on_seam).sum())
        vertices = numpy.vstack(
            [
                numpy.column_stack([back, numpy.zeros(len(back))]),
                numpy.column_stack(
                    [
                        front[~panel.on_seam],
                        numpy.full((~panel.on_seam).sum(), self.layer_gap),
                    ]
                ),
            ]
        )
        vertices[:, :2] -= (
            vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0)
        ) / 2
        return SpreadGarmentMesh(
            surface=trimesh.Trimesh(
                vertices=vertices,
                faces=numpy.vstack(
                    [panel.triangles[:, ::-1], front_vertex[panel.triangles]]
                ),
                process=False,
            ),
            landmark_vertices=panel.landmark_vertices,
        )

    @cached_property
    def _panel(self) -> _TrousersPanel:
        """
        One panel of the trousers, its rows of vertices running across it from the
        waistband down to the crotch, and on down each leg to its cuff.
        """
        crotch_height = self.length - self.rise
        half_waist = self.waist_width / 2
        outseam_end = self.cuff_gap / 2 + self.cuff_width
        column_count = max(1, round(half_waist / self.edge_length))
        rise_row_count = max(1, round(self.rise / self.edge_length))
        leg_row_count = max(1, round(crotch_height / self.edge_length))

        def right_outseam(height: float) -> float:
            """
            :param height: How far above the cuffs.
            :return: Where the right outseam lies along x at that height.
            """
            return -(outseam_end + (half_waist - outseam_end) * height / self.length)

        def right_inseam(height: float) -> float:
            """
            :param height: How far above the cuffs, no higher than the crotch.
            :return: Where the right inseam lies along x at that height.
            """
            return -self.cuff_gap / 2 * (1 - height / crotch_height)

        positions = []

        def added_row(start: float, end: float, height: float, count: int) -> list[int]:
            """
            :param start: Where the row starts along x.
            :param end: Where the row ends along x.
            :param height: How far above the cuffs the row lies.
            :param count: How many vertices the row has.
            :return: The indices of the row's vertices, which are added evenly spaced.
            """
            first = len(positions)
            positions.extend([x, height] for x in numpy.linspace(start, end, count))
            return list(range(first, first + count))

        rise_rows = [
            added_row(
                right_outseam(height),
                -right_outseam(height),
                height,
                2 * column_count + 1,
            )
            for height in numpy.linspace(self.length, crotch_height, rise_row_count + 1)
        ]
        leg_heights = numpy.linspace(crotch_height, 0.0, leg_row_count + 1)[1:]
        right_leg_rows = [rise_rows[-1][: column_count + 1]] + [
            added_row(
                right_outseam(height), right_inseam(height), height, column_count + 1
            )
            for height in leg_heights
        ]
        left_leg_rows = [rise_rows[-1][column_count:]] + [
            added_row(
                -right_inseam(height), -right_outseam(height), height, column_count + 1
            )
            for height in leg_heights
        ]
        triangles = []
        for rows in (rise_rows, right_leg_rows, left_leg_rows):
            for row_index, (upper, lower) in enumerate(zip(rows, rows[1:])):
                for column in range(len(upper) - 1):
                    a, b = upper[column], upper[column + 1]
                    d, c = lower[column], lower[column + 1]
                    if (row_index + column) % 2 == 0:
                        triangles.extend([[a, d, c], [a, c, b]])
                    else:
                        triangles.extend([[a, d, b], [b, d, c]])
        crotch = rise_rows[-1][column_count]
        on_seam = numpy.zeros(len(positions), dtype=bool)
        for row in rise_rows + right_leg_rows + left_leg_rows:
            on_seam[[row[0], row[-1]]] = True
        on_seam[crotch] = True
        on_edge = on_seam.copy()
        for row in (rise_rows[0], right_leg_rows[-1], left_leg_rows[-1]):
            on_edge[row] = True
        return _TrousersPanel(
            positions=numpy.array(positions),
            triangles=numpy.array(triangles),
            on_seam=on_seam,
            on_edge=on_edge,
            landmark_vertices=[
                rise_rows[0][0],
                right_leg_rows[-1][0],
                right_leg_rows[-1][-1],
                crotch,
                left_leg_rows[-1][0],
                left_leg_rows[-1][-1],
                rise_rows[0][-1],
            ],
        )
