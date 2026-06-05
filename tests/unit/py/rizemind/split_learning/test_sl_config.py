import pytest
from pydantic import ValidationError
from rizemind.split_learning.config import SplitLearningConfig

# ---------------------------------------------------------------------------
# Construction — valid cases
# ---------------------------------------------------------------------------


def test_minimal_construction():
    cfg = SplitLearningConfig(cut_layer=3)
    assert cfg.cut_layer == 3
    assert cfg.num_rounds_per_step == 2


def test_explicit_num_rounds():
    cfg = SplitLearningConfig(cut_layer=0, num_rounds_per_step=4)
    assert cfg.num_rounds_per_step == 4


def test_cut_layer_zero_is_valid():
    cfg = SplitLearningConfig(cut_layer=0)
    assert cfg.cut_layer == 0


def test_num_rounds_per_step_one_is_valid():
    cfg = SplitLearningConfig(cut_layer=1, num_rounds_per_step=1)
    assert cfg.num_rounds_per_step == 1


# ---------------------------------------------------------------------------
# Validators — invalid cases
# ---------------------------------------------------------------------------


def test_negative_cut_layer_raises():
    with pytest.raises(ValidationError, match="cut_layer must be >= 0"):
        SplitLearningConfig(cut_layer=-1)


def test_zero_num_rounds_raises():
    with pytest.raises(ValidationError, match="num_rounds_per_step must be >= 1"):
        SplitLearningConfig(cut_layer=2, num_rounds_per_step=0)


def test_negative_num_rounds_raises():
    with pytest.raises(ValidationError, match="num_rounds_per_step must be >= 1"):
        SplitLearningConfig(cut_layer=2, num_rounds_per_step=-3)


# ---------------------------------------------------------------------------
# Flower ConfigRecord round-trip (via BaseConfig.to_config_record)
# ---------------------------------------------------------------------------


def test_to_config_record_keys():
    cfg = SplitLearningConfig(cut_layer=5, num_rounds_per_step=2)
    record = cfg.to_config_record()
    # ConfigRecord is flat with dot-delimited keys; for a flat model the keys
    # are just the field names
    assert "cut_layer" in record
    assert "num_rounds_per_step" in record


def test_to_config_record_values():
    cfg = SplitLearningConfig(cut_layer=7, num_rounds_per_step=3)
    record = cfg.to_config_record()
    assert record["cut_layer"] == 7
    assert record["num_rounds_per_step"] == 3


def test_to_config_record_round_trip():
    """Values survive a model_dump → ConfigRecord → dict round-trip."""
    from rizemind.configuration.transform import unflatten

    cfg = SplitLearningConfig(cut_layer=2, num_rounds_per_step=2)
    record = cfg.to_config_record()
    rebuilt = SplitLearningConfig(**unflatten(dict(record)))
    assert rebuilt == cfg
