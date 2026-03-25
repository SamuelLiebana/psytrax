from importlib.metadata import metadata

import numpy as np
import pytest

import psytrax
from psytrax.models.logistic import N_PARAMS, default_E0, log_lik_trial


def test_package_metadata_declares_runtime_dependencies():
    requires = metadata("psytrax").get_all("Requires-Dist") or []
    for dep in ("jax", "numpy", "scipy", "tqdm"):
        assert any(req.lower().startswith(dep) for req in requires)


def test_logistic_fit_smoke_is_quiet(capsys):
    n_trials = 20
    contrast = np.linspace(-1.0, 1.0, n_trials)
    responses = (contrast > 0).astype(float)

    result = psytrax.fit(
        data={"inputs": {"c": contrast}, "responses": responses},
        log_lik_trial=log_lik_trial,
        n_params=N_PARAMS,
        E0=default_E0(n_trials),
        hess_calc=None,
        verbose=False,
    )

    captured = capsys.readouterr()
    combined_output = f"{captured.out}\n{captured.err}"
    assert "psytrax:" not in combined_output
    assert "Fitting" not in combined_output
    assert "MAP" not in combined_output
    assert result["params"].shape == (N_PARAMS, n_trials)
    assert np.isfinite(result["log_evidence"])


def test_fit_rejects_mismatched_trial_lengths():
    with pytest.raises(ValueError, match="responses have 2"):
        psytrax.fit(
            data={
                "inputs": {"c": np.array([0.1, -0.2, 0.3])},
                "responses": np.array([1.0, 0.0]),
            },
            log_lik_trial=log_lik_trial,
            n_params=N_PARAMS,
            hess_calc=None,
            verbose=False,
        )


def test_shared_sigma_fit_smoke():
    n_trials = 20
    contrast = np.linspace(-1.0, 1.0, n_trials)
    responses = (contrast > 0).astype(float)

    result = psytrax.fit(
        data={"inputs": {"c": contrast}, "responses": responses},
        log_lik_trial=log_lik_trial,
        n_params=N_PARAMS,
        E0=default_E0(n_trials),
        shared_sigma=True,
        hess_calc=None,
        verbose=False,
    )

    assert np.isscalar(result["hyper"]["sigma"])
    assert np.isfinite(result["log_evidence"])


def test_fit_rejects_non_positive_times():
    with pytest.raises(ValueError, match="strictly positive"):
        psytrax.fit(
            data={
                "inputs": {"c": np.array([0.1, -0.2])},
                "responses": np.array([1.0, 0.0]),
                "times": np.array([0.0, 0.4]),
            },
            log_lik_trial=log_lik_trial,
            n_params=N_PARAMS,
            hess_calc=None,
            verbose=False,
        )


def test_fit_rejects_session_length_mismatch():
    with pytest.raises(ValueError, match="session_lengths sum to 3"):
        psytrax.fit(
            data={
                "inputs": {"c": np.array([0.1, -0.2])},
                "responses": np.array([1.0, 0.0]),
                "session_lengths": np.array([1, 2]),
            },
            log_lik_trial=log_lik_trial,
            n_params=N_PARAMS,
            hess_calc=None,
            verbose=False,
        )
