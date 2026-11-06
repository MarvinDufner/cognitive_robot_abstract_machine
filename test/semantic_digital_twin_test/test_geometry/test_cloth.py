"""
Tests for the flat patterns cloth is cut from.
"""

from __future__ import annotations

import numpy
import pytest

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.cloth import TShirtPattern

# %% T-shirt pattern


@pytest.fixture
def pattern() -> TShirtPattern:
    """
    :return: A T-shirt pattern of the default size.
    """
    return TShirtPattern()


def test_a_t_shirt_pattern_covers_its_body_and_sleeves_less_the_neck(
    pattern: TShirtPattern,
) -> None:
    """
    The surface covers the body and both sleeves, without the neck opening.
    """
    surface = pattern.create_surface(HomogeneousTransformationMatrix())

    body = pattern.body_width * pattern.body_length
    sleeves = 2 * pattern.sleeve_length * pattern.sleeve_width
    neck = pattern.neck_width * pattern.neck_depth
    assert surface.area == pytest.approx(body + sleeves - neck)


def test_a_t_shirt_pattern_spans_its_sleeves_across_and_its_body_lengthwise(
    pattern: TShirtPattern,
) -> None:
    """
    The surface is flat, spans cuff to cuff along x and hem to shoulders along y.
    """
    surface = pattern.create_surface(HomogeneousTransformationMatrix())

    across, lengthwise, normal = surface.extents
    assert across == pytest.approx(pattern.body_width + 2 * pattern.sleeve_length)
    assert lengthwise == pytest.approx(pattern.body_length)
    assert normal == pytest.approx(0.0)


def test_a_t_shirt_pattern_lies_centred_where_it_is_placed_with_its_sleeves_along_x(
    pattern: TShirtPattern,
) -> None:
    """
    A pattern placed at a pose is centred on it, its sleeves along the pose's x-axis.
    """
    centre = numpy.array([1.0, -2.0, 0.8])
    world_T_pattern = HomogeneousTransformationMatrix.from_xyz_rpy(
        *centre, yaw=numpy.pi / 2
    )

    surface = pattern.create_surface(world_T_pattern)

    across = pattern.body_width + 2 * pattern.sleeve_length
    assert numpy.allclose(surface.bounds.mean(axis=0), centre)
    assert numpy.allclose(surface.extents, [pattern.body_length, across, 0.0])
