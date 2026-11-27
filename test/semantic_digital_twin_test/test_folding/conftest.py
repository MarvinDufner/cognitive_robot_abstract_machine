"""
A T-shirt drawn with straight edges, whose folds can be worked out by hand.
"""

from __future__ import annotations

import numpy
import pytest

from semantic_digital_twin.folding.t_shirt import TShirtTemplate
from semantic_digital_twin.semantic_annotations.garments import TShirtLandmark

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
