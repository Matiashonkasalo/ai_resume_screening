from src.models import build_model
import pytest


def test_invalid_model():
    with pytest.raises(ValueError):
        build_model("invalid_model", {})


def test_build_random_forest():
    model = build_model("random_forest", params={}, class_weight="balanced")

    assert "model" in model.named_steps
