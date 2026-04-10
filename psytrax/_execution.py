"""Execution-strategy selection for JAX backends and precisions.

The goal is to choose the fastest safe path for the user's hardware:

- NVIDIA CUDA: use GPU with float64 throughout.
- Apple Metal: use GPU float32 for MAP optimisation, then CPU float64 for
  Hessian / Laplace evidence.
- Everything else: use CPU float64.

Backend discovery can hard-crash the current process on some systems
(notably Apple Metal), so probing is done in a subprocess and cached.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache

import jax


@dataclass(frozen=True)
class ExecutionPlan:
    name: str
    jax_platforms: str
    map_backend: str
    map_precision: str
    evidence_backend: str
    evidence_precision: str
    description: str

    def as_dict(self):
        return asdict(self)


_CONFIGURED_JAX_PLATFORMS = None


def resolve_execution_plan(device="auto", precision="float64", verbose=False):
    """Return the safest fast execution plan for the requested hardware mode."""
    if device not in ("auto", "cpu", "gpu", "tpu"):
        raise ValueError(f"device must be one of auto/cpu/gpu/tpu, got {device!r}")
    if precision not in ("float32", "float64"):
        raise ValueError(f"precision must be 'float32' or 'float64', got {precision!r}")

    if device == "cpu":
        return _cpu_plan(precision)

    if device == "tpu":
        warnings.warn("TPU execution is not yet optimized in psytrax; falling back to CPU.")
        return _cpu_plan("float64")

    if device == "gpu":
        if _probe_cuda_float64():
            return _cuda_plan(precision)
        if _supports_apple_metal():
            if precision == "float64":
                warnings.warn(
                    "Apple Metal does not support float64. Using hybrid execution: "
                    "Metal float32 MAP optimisation with CPU float64 evidence.",
                    stacklevel=3,
                )
            return _metal_hybrid_plan()
        warnings.warn("No supported GPU backend was detected; falling back to CPU.", stacklevel=3)
        return _cpu_plan("float64")

    # device == "auto"
    if _probe_cuda_float64():
        return _cuda_plan("float64")
    if _supports_apple_metal():
        return _metal_hybrid_plan()
    return _cpu_plan("float64")


def configure_execution_plan(plan: ExecutionPlan):
    """Pin JAX platform discovery before the first backend is initialized."""
    global _CONFIGURED_JAX_PLATFORMS

    if _CONFIGURED_JAX_PLATFORMS is None:
        if "JAX_PLATFORMS" not in os.environ:
            os.environ["JAX_PLATFORMS"] = plan.jax_platforms
        try:
            jax.config.update("jax_platforms", os.environ.get("JAX_PLATFORMS", plan.jax_platforms))
        except Exception:
            # If the runtime was already initialized elsewhere, JAX may reject
            # reconfiguration. We tolerate that and rely on the active runtime.
            pass
        _CONFIGURED_JAX_PLATFORMS = os.environ.get("JAX_PLATFORMS", plan.jax_platforms)
        return

    configured = {p.strip() for p in _CONFIGURED_JAX_PLATFORMS.split(",") if p.strip()}
    requested = {p.strip() for p in plan.jax_platforms.split(",") if p.strip()}
    if not requested.issubset(configured):
        warnings.warn(
            "JAX platforms were already initialized for "
            f"{_CONFIGURED_JAX_PLATFORMS!r}; cannot switch to {plan.jax_platforms!r} "
            "within the same Python process.",
            stacklevel=3,
        )


def fallback_execution_plans(plan: ExecutionPlan):
    """Conservative fallback plans to try after a retryable failure."""
    if plan.name == "metal_hybrid":
        return [_cpu_plan("float64")]
    if plan.name == "cuda_float64":
        return [_cpu_plan("float64")]
    return []


def _cpu_plan(precision):
    return ExecutionPlan(
        name=f"cpu_{precision}",
        jax_platforms="cpu",
        map_backend="cpu",
        map_precision=precision,
        evidence_backend="cpu",
        evidence_precision="float64" if precision == "float64" else precision,
        description=f"CPU {precision}",
    )


def _cuda_plan(precision):
    return ExecutionPlan(
        name=f"cuda_{precision}",
        jax_platforms="cuda,cpu",
        map_backend="gpu",
        map_precision=precision,
        evidence_backend="gpu",
        evidence_precision="float64" if precision == "float64" else precision,
        description=f"CUDA GPU {precision}",
    )


def _metal_hybrid_plan():
    return ExecutionPlan(
        name="metal_hybrid",
        jax_platforms="metal,cpu",
        map_backend="gpu",
        map_precision="float32",
        evidence_backend="cpu",
        evidence_precision="float64",
        description="Apple Metal float32 MAP + CPU float64 evidence",
    )


@lru_cache(maxsize=None)
def _probe_cuda_float64():
    return _probe_backend("cuda,cpu", "gpu", require_float64=True)


@lru_cache(maxsize=None)
def _supports_apple_metal():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    return _probe_backend("metal,cpu", "gpu", require_float64=False)


def _probe_backend(jax_platforms, backend, require_float64):
    code = f"""
import jax
import jax.numpy as jnp
devs = jax.devices({backend!r})
assert devs, "no devices"
if {require_float64!r}:
    jax.device_put(jnp.array(1.0, dtype=jnp.float64), devs[0])
print(devs[0])
"""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = jax_platforms
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0
