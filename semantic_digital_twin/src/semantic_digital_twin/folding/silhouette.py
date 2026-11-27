"""
What a camera looking straight down at a table sees of a cloth lying on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy
import trimesh

from semantic_digital_twin.folding.g_fold import LayeredCloth

# %% silhouettes


@dataclass(kw_only=True)
class ClothSilhouette:
    """
    The cells of a square grid on the table's plane that a cloth covers, seen from
    straight above.
    """

    covered: numpy.ndarray
    """
    Whether the cloth covers each cell, of shape (rows, columns); rows run along the
    plane's y-axis and columns along its x-axis.
    """

    origin: numpy.ndarray
    """
    The centre of the first cell of the first row, on the table's plane.
    """

    cell_size: float
    """
    The length of a side of each cell, in metres.
    """

    @classmethod
    def from_cloth(cls, *, cloth: LayeredCloth, cell_size: float) -> ClothSilhouette:
        """
        :param cloth: Cloth lying flat.
        :param cell_size: The length of a side of each cell, in metres.
        :return: The silhouette of the cloth, on a grid just covering it.
        """
        vertices = numpy.concatenate([layer.vertices for layer in cloth.layers])
        origin = vertices.min(axis=0) - cell_size
        column_count, row_count = (
            numpy.ceil((vertices.max(axis=0) + cell_size - origin) / cell_size).astype(
                int
            )
            + 1
        )
        columns, rows = numpy.meshgrid(
            numpy.arange(column_count), numpy.arange(row_count)
        )
        centres = origin + cell_size * numpy.column_stack(
            [columns.ravel(), rows.ravel()]
        )
        return cls(
            covered=cloth.contains(centres).reshape(row_count, column_count),
            origin=origin,
            cell_size=cell_size,
        )

    @classmethod
    def from_surface(
        cls, *, surface: trimesh.Trimesh, cell_size: float
    ) -> ClothSilhouette:
        """
        :param surface: The triangles of a cloth, with their vertices in the table's
            frame.
        :param cell_size: The length of a side of each cell, in metres.
        :return: The silhouette of the triangles seen from straight above, on a grid
            just covering them.
        """
        triangles = surface.vertices[:, :2][surface.faces]
        origin = triangles.reshape(-1, 2).min(axis=0) - cell_size
        column_count, row_count = (
            numpy.ceil(
                (triangles.reshape(-1, 2).max(axis=0) + cell_size - origin) / cell_size
            ).astype(int)
            + 1
        )
        covered = numpy.zeros((row_count, column_count), dtype=bool)
        for first, second, third in triangles:
            (to_second_x, to_second_y), (to_third_x, to_third_y) = (
                second - first,
                third - first,
            )
            twice_area = to_second_x * to_third_y - to_second_y * to_third_x
            if abs(twice_area) < cell_size**2 * 1e-6:
                continue
            lowest = numpy.floor(
                (numpy.minimum(numpy.minimum(first, second), third) - origin)
                / cell_size
            ).astype(int)
            highest = numpy.ceil(
                (numpy.maximum(numpy.maximum(first, second), third) - origin)
                / cell_size
            ).astype(int)
            columns, rows = numpy.meshgrid(
                numpy.arange(lowest[0], highest[0] + 1),
                numpy.arange(lowest[1], highest[1] + 1),
            )
            offsets = origin + cell_size * numpy.stack([columns, rows], axis=-1) - first
            offset_x, offset_y = offsets[..., 0], offsets[..., 1]
            second_weight = (offset_x * to_third_y - offset_y * to_third_x) / twice_area
            third_weight = (
                to_second_x * offset_y - to_second_y * offset_x
            ) / twice_area
            inside = (
                (second_weight >= 0)
                & (third_weight >= 0)
                & (second_weight + third_weight <= 1)
            )
            covered[rows[inside], columns[inside]] = True
        return cls(covered=covered, origin=origin, cell_size=cell_size)

    @cached_property
    def covered_points(self) -> numpy.ndarray:
        """
        The centre of each covered cell, of shape (n, 2).
        """
        rows, columns = numpy.nonzero(self.covered)
        return self.origin + self.cell_size * numpy.column_stack([columns, rows])

    @cached_property
    def centroid(self) -> numpy.ndarray:
        """
        The centre of the area the cloth covers.
        """
        return self.covered_points.mean(axis=0)

    @property
    def area(self) -> float:
        """
        The area the cloth covers, in square metres.
        """
        return float(self.covered.sum()) * self.cell_size**2

    def covers(self, points: numpy.ndarray) -> numpy.ndarray:
        """
        :param points: Points on the table's plane, of shape (n, 2).
        :return: Whether the cloth covers the cell each point lies in.
        """
        columns, rows = (
            numpy.round((points - self.origin) / self.cell_size).astype(int).T
        )
        on_grid = (
            (rows >= 0)
            & (rows < self.covered.shape[0])
            & (columns >= 0)
            & (columns < self.covered.shape[1])
        )
        covered = numpy.zeros(len(points), dtype=bool)
        covered[on_grid] = self.covered[rows[on_grid], columns[on_grid]]
        return covered
