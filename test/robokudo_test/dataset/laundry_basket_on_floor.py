"""
A synthetic camera frame of a laundry basket holding laundry, seen from straight above.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from robokudo.types.cv import Point2D, Rect

from .table_from_above import TableFromAboveScene

# %% the scene


@dataclass
class LaundryBasketScene(TableFromAboveScene):
    """
    A camera one metre above the floor looks straight down into a square laundry basket.

    Its rim stands higher than the laundry inside, which lies flat but for one peak.
    """

    basket_region: Rect = field(
        default_factory=lambda: Rect.from_corner_points(60, 60, 140, 140)
    )
    """
    The pixels the basket covers, rim included.
    """

    rim_width: int = 6
    """
    Width of the rim around the basket, in pixels.
    """

    rim_depth: int = 600
    """
    Distance from the camera to the top of the rim, in millimetres.
    """

    laundry_depth: int = 800
    """
    Distance from the camera to the flat part of the laundry, in millimetres.
    """

    peak_pixel: Point2D = field(default_factory=lambda: Point2D(x=110, y=90))
    """
    The pixel at which the laundry is highest.
    """

    peak_depth: int = 700
    """
    Distance from the camera to the peak of the laundry, in millimetres.
    """

    basket_gray_level: int = 50
    """
    Brightness of the basket and the laundry in it.
    """

    def color_image(self) -> npt.NDArray[np.uint8]:
        """
        :return: The basket on the floor.
        """
        image = self.table_image()
        left, top, right, bottom = self.basket_region.get_corner_points()
        image[top:bottom, left:right] = self.basket_gray_level
        return image

    def depth_image(self) -> npt.NDArray[np.uint16]:
        """
        :return: The distance to the floor, the rim, the laundry and its peak.
        """
        depth = self.table_depth_image()
        left, top, right, bottom = self.basket_region.get_corner_points()
        depth[top:bottom, left:right] = self.rim_depth
        depth[
            top + self.rim_width : bottom - self.rim_width,
            left + self.rim_width : right - self.rim_width,
        ] = self.laundry_depth
        depth[self.peak_pixel.y, self.peak_pixel.x] = self.peak_depth
        return depth

    def basket_mask(self) -> npt.NDArray[np.uint8]:
        """
        :return: The mask of the whole basket within its region, as segmentation
            delivers it.
        """
        return np.full(
            (self.basket_region.height, self.basket_region.width),
            np.iinfo(np.uint8).max,
            dtype=np.uint8,
        )
