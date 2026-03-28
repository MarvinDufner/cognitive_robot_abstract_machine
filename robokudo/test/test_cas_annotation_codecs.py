import numpy as np

from robokudo.io import cas_annotation_codecs


def test_deserialize_annotations_none_returns_empty_list():
    assert cas_annotation_codecs.deserialize_annotations(None) == []


def test_deserialize_annotations_forwards_kwargs_for_lists(monkeypatch):
    calls: list[tuple[dict, dict]] = []

    def fake_from_json(item, **kwargs):
        calls.append((item, kwargs))
        return {"item": item, "kwargs": kwargs}

    monkeypatch.setattr(cas_annotation_codecs, "from_json", fake_from_json)

    payload = [{"id": 1}, {"id": 2}]
    result = cas_annotation_codecs.deserialize_annotations(payload, tracker="active")

    assert len(result) == 2
    assert [call[0] for call in calls] == payload
    assert all(call[1] == {"tracker": "active"} for call in calls)


def test_deserialize_annotations_forwards_kwargs_for_single_item(monkeypatch):
    calls: list[tuple[dict, dict]] = []

    def fake_from_json(item, **kwargs):
        calls.append((item, kwargs))
        return {"item": item, "kwargs": kwargs}

    monkeypatch.setattr(cas_annotation_codecs, "from_json", fake_from_json)

    payload = {"id": 3}
    result = cas_annotation_codecs.deserialize_annotations(payload, tracker="active")

    assert len(result) == 1
    assert calls == [(payload, {"tracker": "active"})]
    assert result[0]["item"] == payload


def test_numpy_scalar_json_serializer_roundtrip():
    scalar = np.int16(42)
    encoded = cas_annotation_codecs.NumpyScalarJSONSerializer.to_json(scalar)
    restored = cas_annotation_codecs.NumpyScalarJSONSerializer.from_json(
        encoded, np.generic
    )

    assert encoded["dtype"] == "int16"
    assert restored == scalar.item()
