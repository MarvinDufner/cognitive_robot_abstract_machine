"""
Tests for placing the mask of a region of interest on its whole image.
"""

from __future__ import annotations

import numpy as np
import pytest

from robokudo.types.cv import ImageROI, Rect

# %% fixtures


def region_of_interest(left: int, top: int, width: int, height: int) -> ImageROI:
    """
    :return: A region of interest whose mask covers it entirely.
    """
    roi = ImageROI()
    roi.roi = Rect.from_corner_points(left, top, left + width, top + height)
    roi.mask = np.full((height, width), np.iinfo(np.uint8).max, dtype=np.uint8)
    return roi


def expected_mask(
    image_shape: tuple[int, int], left: int, top: int, right: int, bottom: int
) -> np.ndarray:
    """
    :return: An empty image covered from ``left, top`` up to ``right, bottom``.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[top:bottom, left:right] = np.iinfo(np.uint8).max
    return mask


# %% placing the mask


def test_a_region_inside_the_image_is_placed_where_it_lies() -> None:
    """
    The mask covers the region's pixels and nothing else.
    """
    roi = region_of_interest(left=2, top=3, width=4, height=5)

    mask = roi.full_image_mask((10, 10))

    assert (mask == expected_mask((10, 10), 2, 3, 6, 8)).all()


@pytest.mark.parametrize(
    ("left", "top", "expected_corners"),
    [
        (-2, 3, (0, 3, 2, 8)),
        (2, -3, (2, 0, 6, 2)),
        (8, 3, (8, 3, 10, 8)),
        (2, 7, (2, 7, 6, 10)),
    ],
    ids=[
        "past the left edge",
        "past the top edge",
        "past the right edge",
        "past the bottom edge",
    ],
)
def test_a_region_reaching_past_the_image_is_cut_at_its_edge(
    left: int, top: int, expected_corners: tuple[int, int, int, int]
) -> None:
    """
    Detectors may report regions reaching past the image; only the part inside is
    placed.
    """
    roi = region_of_interest(left=left, top=top, width=4, height=5)

    mask = roi.full_image_mask((10, 10))

    assert (mask == expected_mask((10, 10), *expected_corners)).all()
