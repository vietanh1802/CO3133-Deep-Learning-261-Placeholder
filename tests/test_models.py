import pytest

from src.models import Linear, build_model


def test_model_factory_dispatches_to_named_model() -> None:
    model = build_model("linear", {})

    assert isinstance(model, Linear)


def test_model_factory_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("unknown", {})
