"""
Tests for freezing the scene after a query named ``freeze``, so that its result stays on
view until the next query arrives.
"""

from __future__ import annotations

import pytest
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from robokudo_msgs.action import Query

from robokudo.annotators.scene_freeze import (
    FreezeSceneOnRequest,
    HoldFrozenScene,
    SceneFreeze,
    SceneQueryName,
)
from robokudo.cas import CAS, CASViews
from robokudo.identifier import BBIdentifier

from .dataset.waiting_query_server import WaitingQueryServer
from .test_cloth_unfolding import update_in_pipeline

pytestmark = pytest.mark.usefixtures("annotator_output_buffer")

# %% fixtures


@pytest.fixture
def query_server() -> WaitingQueryServer:
    """
    :return: A query server with no query waiting, where the pipeline looks for it.
    """
    server = WaitingQueryServer()
    Blackboard().set(BBIdentifier.QUERY_SERVER, server)
    return server


@pytest.fixture
def frozen_scene() -> SceneFreeze:
    """
    :return: A scene frozen by an earlier query.
    """
    return SceneFreeze(is_frozen=True)


def answered_query(query_name: str) -> CAS:
    """
    :return: A CAS whose query, named as given, has been answered.
    """
    cas = CAS()
    cas.set(CASViews.QUERY, Query.Goal(query=query_name))
    return cas


# %% freezing on request


def test_a_query_named_freeze_freezes_the_scene() -> None:
    """
    Answering a query named ``freeze`` freezes the scene.
    """
    scene_freeze = SceneFreeze()

    update_in_pipeline(
        FreezeSceneOnRequest(scene_freeze=scene_freeze),
        answered_query(SceneQueryName.FREEZE),
    )

    assert scene_freeze.is_frozen


def test_a_query_with_another_name_leaves_the_scene_running(
    frozen_scene: SceneFreeze,
) -> None:
    """
    Any other query keeps the scene updating, even after a frozen one.
    """
    update_in_pipeline(
        FreezeSceneOnRequest(scene_freeze=frozen_scene), answered_query("detect")
    )

    assert not frozen_scene.is_frozen


# %% holding the frozen scene


def test_a_running_scene_is_not_held(query_server: WaitingQueryServer) -> None:
    """
    Without a freeze, every iteration goes on to read the camera.
    """
    status = update_in_pipeline(HoldFrozenScene(scene_freeze=SceneFreeze()), CAS())

    assert status == Status.SUCCESS


def test_a_frozen_scene_is_held_while_no_query_waits(
    frozen_scene: SceneFreeze, query_server: WaitingQueryServer
) -> None:
    """
    A frozen scene keeps the last result on view until the next query.
    """
    status = update_in_pipeline(HoldFrozenScene(scene_freeze=frozen_scene), CAS())

    assert status == Status.RUNNING
    assert frozen_scene.is_frozen


def test_the_next_query_thaws_the_scene_without_taking_the_query(
    frozen_scene: SceneFreeze, query_server: WaitingQueryServer
) -> None:
    """
    The next query lets the pipeline run again, and stays waiting for it to answer.
    """
    query = Query.Goal(query="detect")
    query_server.new_query = query

    status = update_in_pipeline(HoldFrozenScene(scene_freeze=frozen_scene), CAS())

    assert status == Status.SUCCESS
    assert not frozen_scene.is_frozen
    assert query_server.new_query is query
