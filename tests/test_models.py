import numpy as np

from psytrax.models.ddm import log_lik_trial as ddm_log_lik_trial
from psytrax.models.race import (
    default_model_hyper_with_dopamine,
    _log_lik_dopamine,
    log_lik_trial as race_log_lik_trial,
)


def test_rt_models_return_finite_penalty_for_invalid_times_and_thresholds():
    race_invalid_time = race_log_lik_trial(
        np.array([1.0, 1.0, 0.5, 0.5, 1.0, 0.1]),
        {"inputs": {"c": 1.0}, "r": 1.0, "T": 0.0},
    )
    race_invalid_threshold = race_log_lik_trial(
        np.array([1.0, 1.0, 0.5, 0.5, -1.0, 0.1]),
        {"inputs": {"c": 1.0}, "r": 1.0, "T": 0.5},
    )
    ddm_invalid_time = ddm_log_lik_trial(
        np.array([1.0, 0.0, 1.0]),
        {"inputs": {"c": 1.0}, "r": 1.0, "T": -0.1},
    )
    ddm_invalid_threshold = ddm_log_lik_trial(
        np.array([1.0, 0.0, -1.0]),
        {"inputs": {"c": 1.0}, "r": 1.0, "T": 0.5},
    )

    for value in (
        race_invalid_time,
        race_invalid_threshold,
        ddm_invalid_time,
        ddm_invalid_threshold,
    ):
        assert np.isfinite(np.asarray(value))
        assert float(value) < -1e6


def test_race_dopamine_default_model_hyper_starts_near_fitted_region():
    model_hyper = default_model_hyper_with_dopamine()

    assert model_hyper["sig_i"] == 0.01
    assert model_hyper["da_beta"] == 2.0
    assert model_hyper["da_offset"] == 0.001


def test_race_dopamine_tanh_readout_is_zero_centered():
    params = np.array([1.0, 1.0, 0.8, 0.8, 1.0])
    model_hyper = {
        "sig_i": 0.01,
        "sig_DA": 0.1,
        "da_beta": 2.0,
        "da_offset": 0.0,
    }
    dat = {"inputs": {"c": 0.0}, "r": 1.0, "T": 0.5}

    ll_at_zero = race_log_lik_trial(
        params,
        {**dat, "dopamine": 0.0},
        model_hyper,
    )
    ll_at_unshifted_midpoint = race_log_lik_trial(
        params,
        {**dat, "dopamine": 0.5},
        model_hyper,
    )

    assert float(np.asarray(ll_at_zero)) > float(np.asarray(ll_at_unshifted_midpoint))


def test_race_dopamine_readout_uses_threshold_normalised_drive():
    model_hyper = {
        "sig_DA": 0.1,
        "da_beta": 2.0,
        "da_offset": 0.0,
    }
    dat = {"inputs": {"c": 1.0}, "dopamine": np.tanh(1.0)}
    ll_z1 = _log_lik_dopamine(
        np.array([1.0, 1.0, 0.8, 0.8, 1.0]),
        dat,
        model_hyper,
    )
    ll_z2 = _log_lik_dopamine(
        np.array([1.0, 1.0, 0.8, 0.8, 2.0]),
        dat,
        model_hyper,
    )

    assert float(np.asarray(ll_z1)) > float(np.asarray(ll_z2))
