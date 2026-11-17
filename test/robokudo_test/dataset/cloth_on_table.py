"""
A synthetic camera frame of a cloth lying on a table, seen from straight above.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from robokudo.types.cv import Rect

from .table_from_above import TableFromAboveScene

# %% the scene


@dataclass
class ClothOnTableScene(TableFromAboveScene):
    """
    A camera one metre above a table looks straight down at a bright, rectangular cloth
    whose surface rises by a millimetre per pixel towards the right of the image.
    """

    cloth_region: Rect = field(
        default_factory=lambda: Rect.from_corner_points(110, 110, 190, 190)
    )
    """
    The pixels the cloth covers.
    """

    cloth_gray_level: int = 200
    """
    Brightness of the cloth.
    """

    def color_image(self) -> npt.NDArray[np.uint8]:
        """
        :return: The cloth on the table.
        """
        image = self.table_image()
        left, top, right, bottom = self.cloth_region.get_corner_points()
        image[top:bottom, left:right] = self.cloth_gray_level
        return image

    def depth_image(self) -> npt.NDArray[np.uint16]:
        """
        :return: The distance to the table, and to the cloth rising towards the right.
        """
        depth = self.table_depth_image()
        left, top, right, bottom = self.cloth_region.get_corner_points()
        depth[top:bottom, left:right] = self.table_depth - (
            np.arange(left, right) - left
        )
        return depth

    def contour(self) -> npt.NDArray[np.int32]:
        """
        :return: The outline of the cloth, as OpenCV traces it.
        """
        left, top, right, bottom = self.cloth_region.get_corner_points()
        return np.array(
            [
                [[left, top]],
                [[left, bottom - 1]],
                [[right - 1, bottom - 1]],
                [[right - 1, top]],
            ],
            dtype=np.int32,
        )
