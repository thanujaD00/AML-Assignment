"""Sanity tests for data integrity, leakage-free features and the evaluation split."""

import numpy as np
import pandas as pd
import pytest

from dengue import config
from dengue.data import load_data, quality_report
from dengue.evaluation import rolling_origin_predict
from dengue.features import build_supervised, growth_to_cases
from dengue.models import PersistenceForecaster


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def sup(data):
    return build_supervised(data)


def test_raw_shape_and_places(data):
    assert len(data.raw) == 1242
    assert len(data.divisions) == 26
    assert data.weekly["date"].nunique() == 10


def test_divisions_sum_to_national(data):
    q = quality_report(data)
    assert (q["national_sum_check"]["max_abs_gap_divisions_vs_national"] == 0).all()
    assert (q["ytd_consistency_check"]["max_abs_gap_ytd_increment_vs_week"] == 0).all()


def test_unreliable_weeks_never_used(sup):
    bad = set(config.UNRELIABLE_DENGUE_WEEKS)
    assert not sup["origin_date"].isin(bad).any()
    assert not sup["target_date"].isin(bad).any()


def test_supervised_table_size(sup):
    assert len(sup) == 6 * 26
    assert sup[config.FEATURES + ["target_log_growth"]].notna().all().all()


def test_target_is_next_week(sup, data):
    w = data.weekly.set_index(["place", "date"])["dengue_week"]
    r = sup.iloc[10]
    assert r.target_date - r.origin_date == pd.Timedelta(days=7)
    assert w[(r.place, r.target_date)] == r.cases_next
    assert w[(r.place, r.origin_date)] == r.cases_t


def test_no_future_rows_in_training(sup):
    seen = {}

    class Spy(PersistenceForecaster):
        def fit(self, X, y=None):
            seen["n"] = len(X)
            return self

    origin = config.TEST_ORIGINS[0]
    rolling_origin_predict(sup, Spy, [origin])
    assert seen["n"] == (sup["origin_date"] < origin).sum()
    # every training target week is <= the forecast origin
    assert (sup.loc[sup.origin_date < origin, "target_date"] <= origin).all()


def test_growth_to_cases_roundtrip():
    assert np.allclose(growth_to_cases([100.0], [0.0]), [100.0])
    assert growth_to_cases([10.0], [-50.0])[0] >= 0
