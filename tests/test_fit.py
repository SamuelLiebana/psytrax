from importlib.metadata import metadata

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from scipy.optimize import OptimizeResult
import jax.numpy as jnp

import psytrax
import psytrax._execution as execution_mod
from psytrax._helper.helperFunctions import myblk_diags, sparse_logdet
from psytrax._hyper_opt import (
    _block_tridiag_solve_logdet_jax,
    _blocks_from_prior_and_likelihood,
    _hyperparameter_update_size,
    _limit_hyperparameter_step,
    _should_retry_hyper_minimize,
    _sparse_trial_blocks,
)
from psytrax._jax_map import _raise_if_invalid_solution
from psytrax.fit import _is_retryable_fit_error, _model_default_E0
from psytrax.models.logistic import N_PARAMS, default_E0, log_lik_trial


def test_package_metadata_declares_runtime_dependencies():
    requires = metadata("psytrax").get_all("Requires-Dist") or []
    for dep in ("jax", "optax", "numpy", "scipy", "tqdm"):
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


def test_invalid_fit_solution_raises_clear_error():
    with pytest.raises(RuntimeError, match="invalid parameter region"):
        _raise_if_invalid_solution(-1e12, 100)


def test_plausible_fit_solution_passes_validation():
    _raise_if_invalid_solution(-50.0, 100)


def test_retryable_fit_error_detection():
    assert _is_retryable_fit_error(RuntimeError("invalid parameter region"))
    assert _is_retryable_fit_error(RuntimeError("non-finite log-evidence"))
    assert _is_retryable_fit_error(RuntimeError("Posterior Hessian remained exactly singular"))
    assert not _is_retryable_fit_error(ValueError("responses must be finite"))


def test_sparse_logdet_regularizes_singular_matrix():
    mat = csc_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
    value = sparse_logdet(mat)
    assert np.isfinite(value)


def test_hyper_minimize_retry_detects_nonstationary_stall():
    x0 = np.array([-1.0, -2.0])
    result = OptimizeResult(x=x0.copy(), jac=np.array([12.0, 0.0]), fun=10.0)

    assert _should_retry_hyper_minimize(result, x0, [(-15, 5), (-15, 5)])


def test_hyper_minimize_retry_ignores_real_step_or_tiny_gradient():
    x0 = np.array([-1.0, -2.0])
    moved = OptimizeResult(x=np.array([-1.2, -2.0]), jac=np.array([12.0, 0.0]), fun=9.0)
    stationary = OptimizeResult(x=x0.copy(), jac=np.array([1e-4, 0.0]), fun=10.0)

    assert not _should_retry_hyper_minimize(moved, x0, [(-15, 5), (-15, 5)])
    assert not _should_retry_hyper_minimize(stationary, x0, [(-15, 5), (-15, 5)])


def test_hyperparameter_update_size_handles_log2_zero_start():
    old_x = np.array([1.0, 0.0])
    new_x = np.array([1.0, 0.01])

    assert _hyperparameter_update_size(old_x, new_x) == pytest.approx(0.01)


def test_hyperparameter_step_limiter_rejects_non_improving_large_step(monkeypatch):
    x0 = np.array([0.0])
    result = OptimizeResult(x=np.array([10.0]), fun=0.0)
    result.psytrax_method = "test"

    import psytrax._hyper_opt as hyper_opt_mod

    monkeypatch.setattr(
        hyper_opt_mod,
        "_hyperOpt_lossfun",
        lambda x, _keywords: 1.0 + float(np.sum(np.asarray(x) ** 2)),
    )

    limited = _limit_hyperparameter_step(
        result, x0, {}, max_log2_step=2.0,
    )

    assert np.allclose(limited.x, x0)
    assert "step rejected" in limited.psytrax_method


def test_jax_block_helpers_match_sparse_hessian_layout_and_logdet():
    n_trials = 4
    n_params = 2
    blocks = np.array([
        [[-1.0, 0.1], [0.1, -0.8]],
        [[-1.2, 0.0], [0.0, -0.7]],
        [[-0.9, 0.2], [0.2, -1.1]],
        [[-1.1, 0.1], [0.1, -0.6]],
    ])
    sparse_h = myblk_diags(blocks)

    recovered = _sparse_trial_blocks(sparse_h, n_params, n_trials)
    assert np.allclose(recovered, blocks)

    q_diag = jnp.asarray(np.full((n_trials, n_params), 2.0))
    q_off = jnp.asarray(np.full((n_trials - 1, n_params), -0.25))
    dense = np.zeros((n_trials * n_params, n_trials * n_params))
    for t in range(n_trials):
        dense[t * n_params:(t + 1) * n_params,
              t * n_params:(t + 1) * n_params] = (
            np.diag(np.asarray(q_diag[t])) - blocks[t]
        )
        if t + 1 < n_trials:
            off = np.diag(np.asarray(q_off[t]))
            dense[t * n_params:(t + 1) * n_params,
                  (t + 1) * n_params:(t + 2) * n_params] = off
            dense[(t + 1) * n_params:(t + 2) * n_params,
                  t * n_params:(t + 1) * n_params] = off

    A = _blocks_from_prior_and_likelihood(jnp, q_diag, jnp.asarray(blocks))
    _, logdet = _block_tridiag_solve_logdet_jax(
        jnp, A, q_off, jnp.zeros((n_trials, n_params)), solve_rhs=False,
    )
    assert np.allclose(float(logdet), np.linalg.slogdet(dense)[1])


def test_model_default_e0_is_discovered_for_builtin_model():
    E0 = _model_default_E0(log_lik_trial, 12, N_PARAMS)
    assert E0.shape == (N_PARAMS, 12)


def test_auto_execution_prefers_metal_hybrid_when_cuda_absent(monkeypatch):
    monkeypatch.setattr(execution_mod, "_probe_cuda_float64", lambda: False)
    monkeypatch.setattr(execution_mod, "_supports_apple_metal", lambda: True)
    plan = execution_mod.resolve_execution_plan(device="auto", precision="float64")
    assert plan.name == "metal_hybrid"
    assert plan.map_precision == "float32"
    assert plan.evidence_precision == "float64"


def test_auto_execution_prefers_cpu_when_no_gpu_backend(monkeypatch):
    monkeypatch.setattr(execution_mod, "_probe_cuda_float64", lambda: False)
    monkeypatch.setattr(execution_mod, "_supports_apple_metal", lambda: False)
    plan = execution_mod.resolve_execution_plan(device="auto", precision="float64")
    assert plan.name == "cpu_float64"
