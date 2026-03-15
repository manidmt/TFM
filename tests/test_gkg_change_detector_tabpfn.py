"""Tests for TabPFN support in the GKG change detector."""
import sys
import unittest.mock as mock

import numpy as np
import pytest

from quant_risk.models.tabular.gkg_change_detector import (
    GkgChangeDetectorConfig,
    fit,
    make_model,
    predict_proba,
)


def test_tabpfn_config_defaults():
    cfg = GkgChangeDetectorConfig(model_type="tabpfn")
    assert cfg.tabpfn_n_estimators == 8
    assert cfg.tabpfn_softmax_temperature == pytest.approx(0.9)
    assert cfg.tabpfn_balance_probabilities is False
