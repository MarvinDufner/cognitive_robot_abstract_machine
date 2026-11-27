"""
A T-shirt and a pair of trousers drawn with straight edges, whose folds can be worked
out by hand.
"""

from __future__ import annotations

import numpy
import pytest

from semantic_digital_twin.folding.t_shirt import TShirtTemplate
from semantic_digital_twin.folding.trousers import TrousersTemplate
from semantic_digital_twin.semantic_annotations.garments import (
    TShirtLandmark,
    TrousersLandmark,
)

# %% the T-shirt


@pytest.fixture(scope="session")
def t_shirt_template() -> TShirtTemplate:
    """
    :return: A T-shirt whose body is 0.5 m wide, its hem on the x-axis, its shoulders
        0.6 m above the hem and sloping up to its collar 0.7 m above the hem, with a
        0.15 m long and 0.2 m wide rectangular sleeve on either side below the
        shoulders. Its wearer's right is on the negative side of the x-axis.
    """
    return TShirtTemplate(
        landmark_positions={
            TShirtLandmark.RIGHT_HEM: numpy.array([-0.25, 0.0]),
            TShirtLandmark.RIGHT_ARMPIT: numpy.array([-0.25, 0.4]),
            TShirtLandmark.RIGHT_SLEEVE_BOTTOM: numpy.array([-0.4, 0.4]),
            TShirtLandmark.RIGHT_SLEEVE_TOP: numpy.array([-0.4, 0.6]),
            TShirtLandmark.RIGHT_SHOULDER: numpy.array([-0.25, 0.6]),
            TShirtLandmark.RIGHT_COLLAR: numpy.array([-0.08, 0.7]),
            TShirtLandmark.LEFT_COLLAR: numpy.array([0.08, 0.7]),
            TShirtLandmark.LEFT_SHOULDER: numpy.array([0.25, 0.6]),
            TShirtLandmark.LEFT_SLEEVE_TOP: numpy.array([0.4, 0.6]),
            TShirtLandmark.LEFT_SLEEVE_BOTTOM: numpy.array([0.4, 0.4]),
            TShirtLandmark.LEFT_ARMPIT: numpy.array([0.25, 0.4]),
            TShirtLandmark.LEFT_HEM: numpy.array([0.25, 0.0]),
        }
    )


# %% the trousers


@pytest.fixture(scope="session")
def trousers_template() -> TrousersTemplate:
    """
    :return: Trousers 1 m long, their 0.4 m wide waistband 1 m above the cuffs, which
        lie on the x-axis, each 0.2 m wide with 0.1 m between them, and the crotch 0.7 m
        above the cuffs. The wearer's right is on the negative side of the x-axis.
    """
    return TrousersTemplate(
        landmark_positions={
            TrousersLandmark.RIGHT_WAIST: numpy.array([-0.2, 1.0]),
            TrousersLandmark.RIGHT_CUFF_OUTSEAM: numpy.array([-0.25, 0.0]),
            TrousersLandmark.RIGHT_CUFF_INSEAM: numpy.array([-0.05, 0.0]),
            TrousersLandmark.CROTCH: numpy.array([0.0, 0.7]),
            TrousersLandmark.LEFT_CUFF_INSEAM: numpy.array([0.05, 0.0]),
            TrousersLandmark.LEFT_CUFF_OUTSEAM: numpy.array([0.25, 0.0]),
            TrousersLandmark.LEFT_WAIST: numpy.array([0.2, 1.0]),
        }
    )
