"""Forward simulation and model recovery for psytrax.

A model in psytrax is fully specified by:
  - a per-trial log-likelihood ``log_lik_trial(params, dat_trial)``  (used by ``psytrax.fit``)
  - a per-trial sampler           ``sample_trial(params, dat_trial, rng)`` (used here)

The sampler is intentionally a plain numpy callable (not JAX) so that users
can simulate from constraint-respecting distributions (e.g. inverse-Gaussian
first-passage times) without having to write a JAX-traceable sampler.  The
overhead of a Python loop over trials is negligible compared to fitting.

Public API
----------
- :func:`simulate` — turn a (K, N) trajectory of true parameters into a data
  dict that can be fed straight into :func:`psytrax.fit`.
- :func:`recover` — convenience wrapper that simulates, fits, and returns
  the fit result with the ground-truth trajectory attached as ``true_params``.

Both functions are model-agnostic: they accept any user-defined sampler /
likelihood pair.  See the built-in models (race, logistic, ddm) for ready-made
``sample_trial`` implementations.
"""

from __future__ import annotations

import numpy as np

from psytrax.fit import fit


def simulate(sample_trial, params, inputs, *,
             rng=None, session_lengths=None, extra_per_trial=None,
             model_hyper=None):
    """Simulate trial-by-trial behaviour from a model.

    Parameters
    ----------
    sample_trial : callable
        Function with signature ``sample_trial(params_t, dat_trial, rng) -> dict``.

        - ``params_t`` is a 1-D numpy array of shape ``(K,)`` holding the
          model parameters at trial *t*.
        - ``dat_trial`` is a dict with keys mirroring the data dict expected by
          :func:`psytrax.fit`.  In particular ``dat_trial['inputs']`` is a
          *scalar-valued* dict for the current trial.  Any keys you put in
          ``extra_per_trial`` are also forwarded as scalars.
        - ``rng`` is a :class:`numpy.random.Generator`.

        The function must return a dict containing at least ``'r'`` (the
        sampled response).  It may also return ``'T'`` (a reaction time) and
        any number of additional scalar fields that you want stored on the
        resulting data dict under a key of the same name (vector across trials).
    params : array-like, shape (K, N)
        Ground-truth parameter trajectory.  Trial *t* uses ``params[:, t]``.
    inputs : dict
        Dict of input arrays with first axis of length ``N``.  Forwarded to
        ``dat_trial['inputs']`` one trial at a time.
    rng : numpy.random.Generator or int, optional
        Random state.  Defaults to ``np.random.default_rng()``.  An ``int`` is
        treated as a seed.
    session_lengths : array-like, optional
        Per-session trial counts.  Stored on the returned dict under the
        ``session_lengths`` key so the simulated data can be fit with
        ``session_boundaries=True``.
    extra_per_trial : dict, optional
        Extra trial-aligned arrays to pass through to ``sample_trial`` as
        scalar fields on ``dat_trial``.  Useful when the sampler depends on
        observable side information that is not a model "input" per se
        (e.g. a reward signal for a learning rule).
    model_hyper : dict, optional
        Model-level scalar hyperparameters forwarded to ``sample_trial`` as
        a fourth positional argument (e.g. the race model's ``sig_i``).
        Defaults to an empty dict.

    Returns
    -------
    dict
        psytrax-formatted data dict with keys:

        - ``inputs``: copy of the user-supplied dict (full arrays).
        - ``responses``: array of shape ``(N,)`` from sampler['r'].
        - ``times``: array of shape ``(N,)`` if the sampler returned ``'T'``.
        - ``session_lengths``: forwarded if provided.
        - any additional scalar fields returned by ``sample_trial``.

    Notes
    -----
    The sampler is called in pure Python (one call per trial).  This is
    intentional — sampling is a one-off cost and using numpy keeps the
    sampler simple to write and debug.
    """
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))

    if model_hyper is None:
        model_hyper = {}

    params = np.asarray(params, dtype=float)
    if params.ndim != 2:
        raise ValueError(f"params must be 2-D (K, N), got shape {params.shape}")
    K, N = params.shape

    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("inputs must be a non-empty dict of trial-aligned arrays")
    inputs_full = {}
    for key, value in inputs.items():
        arr = np.asarray(value)
        if arr.shape[0] != N:
            raise ValueError(
                f"input '{key}' has first dim {arr.shape[0]} but params have N={N}"
            )
        inputs_full[key] = arr

    extra_full = {}
    if extra_per_trial is not None:
        for key, value in extra_per_trial.items():
            arr = np.asarray(value)
            if arr.shape[0] != N:
                raise ValueError(
                    f"extra_per_trial['{key}'] has first dim {arr.shape[0]} "
                    f"but params have N={N}"
                )
            extra_full[key] = arr

    responses = np.empty(N, dtype=float)
    times = None
    other_fields: dict[str, np.ndarray] = {}

    for t in range(N):
        params_t = params[:, t]
        dat_trial = {
            'inputs': {k: v[t] for k, v in inputs_full.items()},
        }
        for k, v in extra_full.items():
            dat_trial[k] = v[t]

        out = sample_trial(params_t, dat_trial, rng, model_hyper)
        if not isinstance(out, dict) or 'r' not in out:
            raise ValueError(
                "sample_trial must return a dict containing at least 'r'"
            )
        responses[t] = float(out['r'])
        if 'T' in out:
            if times is None:
                times = np.empty(N, dtype=float)
            times[t] = float(out['T'])
        for key, value in out.items():
            if key in ('r', 'T'):
                continue
            if key not in other_fields:
                other_fields[key] = np.empty(N, dtype=float)
            other_fields[key][t] = float(value)

    data: dict = {
        'inputs': inputs_full,
        'responses': responses,
    }
    if times is not None:
        data['times'] = times
    if session_lengths is not None:
        data['session_lengths'] = np.asarray(session_lengths)
    for key, arr in other_fields.items():
        data[key] = arr

    return data


def recover(*, sample_trial, log_lik_trial, n_params, true_params, inputs,
            param_names=None, rng=None, session_lengths=None,
            extra_per_trial=None, true_model_hyper=None,
            init_model_hyper=None, **fit_kwargs):
    """Simulate from a model with known parameter trajectories, then fit.

    The result is the standard :func:`psytrax.fit` output dict with one
    extra key:

    - ``true_params`` — the ground-truth ``(K, N)`` trajectory used to
      simulate the data, so callers can directly overlay truth against
      ``result['params']``.

    Parameters
    ----------
    sample_trial : callable
        Per-trial sampler — see :func:`simulate`.
    log_lik_trial : callable
        Per-trial log-likelihood used by :func:`psytrax.fit`.  Must be
        consistent with ``sample_trial`` (i.e. the same model).
    n_params : int
        ``K`` — the number of trial-varying parameters.
    true_params : array-like, shape (K, N)
        Ground-truth parameter trajectory used to simulate the data.
    inputs : dict
        Trial-aligned input arrays — see :func:`simulate`.
    param_names : list[str], optional
        Forwarded to :func:`psytrax.fit`.
    rng : numpy.random.Generator or int, optional
        Random state for the sampler.
    session_lengths : array-like, optional
        Forwarded to both the data dict and the fit (``session_boundaries``
        is *not* set automatically; pass it via ``fit_kwargs`` if desired).
    extra_per_trial : dict, optional
        Extra trial-aligned arrays exposed to the sampler — see
        :func:`simulate`.
    true_model_hyper : dict, optional
        Ground-truth model-level scalar hyperparameters used by the simulator
        (e.g. ``{'sig_i': 0.15}``).  Stored on the result dict as
        ``true_model_hyper`` so callers can compare against
        ``result['model_hyper']`` recovered by EB.
    init_model_hyper : dict, optional
        Starting point for the EB optimisation of ``model_hyper``.  If not
        provided, the model's ``default_model_hyper()`` is used.  Pass this
        explicitly when you want the recovery to start from a value
        *different* from the simulator's truth (the realistic case).
    **fit_kwargs
        Forwarded to :func:`psytrax.fit` (e.g. ``hyper``, ``learning_rule``,
        ``E0``, ``shared_sigma``, ``session_boundaries``, ``device``,
        ``optimise_model_hyper``).

    Returns
    -------
    dict
        ``psytrax.fit`` result dict augmented with ``true_params``,
        ``true_model_hyper``, and ``simulated_data``.
    """
    true_params = np.asarray(true_params, dtype=float)
    if true_params.ndim != 2 or true_params.shape[0] != n_params:
        raise ValueError(
            f"true_params must have shape (n_params={n_params}, N), "
            f"got {true_params.shape}"
        )

    data = simulate(
        sample_trial,
        true_params,
        inputs,
        rng=rng,
        session_lengths=session_lengths,
        extra_per_trial=extra_per_trial,
        model_hyper=true_model_hyper,
    )

    # Caller can either pass init_model_hyper explicitly (typical: a guess that
    # differs from the truth, so recovery has something to do) or let
    # psytrax.fit fall back to the model's default_model_hyper().
    fit_kwargs.setdefault('model_hyper', init_model_hyper)

    result = fit(
        data=data,
        log_lik_trial=log_lik_trial,
        n_params=n_params,
        param_names=param_names,
        **fit_kwargs,
    )

    if not isinstance(result, dict):
        # save=True path — psytrax.fit returned a path; in that case we still
        # want to surface the truth, so load + augment + re-save.
        path = result
        loaded = np.load(path, allow_pickle=True).item()
        loaded['true_params'] = true_params
        loaded['true_model_hyper'] = true_model_hyper or {}
        loaded['simulated_data'] = data
        np.save(path, loaded)
        return path

    result['true_params'] = true_params
    result['true_model_hyper'] = true_model_hyper or {}
    result['simulated_data'] = data
    return result
