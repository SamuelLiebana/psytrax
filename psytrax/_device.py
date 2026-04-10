"""JAX device selection utilities.

GPU installation (float64-capable backends only):
  NVIDIA CUDA 12:  pip install jax[cuda12]
  NVIDIA CUDA 11:  pip install jax[cuda11_pip]

Note: Apple Metal is float32-only. psytrax therefore uses a hybrid strategy
on Apple Silicon when Metal is available: float32 MAP optimisation on Metal,
followed by float64 Hessian / evidence computation on CPU.

After installing a CUDA backend, pass device='gpu' (or 'auto') to psytrax.fit().
"""

import warnings
import jax
import jax.numpy as jnp


def setup_device(device='auto', verbose=True, dtype=jnp.float64):
    """Select a JAX compute device for the requested backend/dtype.

    Args:
        device : 'auto' | 'cpu' | 'gpu' | 'tpu'
            'auto' tries GPU/TPU first and falls back to CPU.
        verbose : bool, print the selected device.
        dtype : JAX dtype that must be placeable on the device.

    Returns:
        The JAX device object that was selected.
    """
    backends = ['gpu', 'tpu', 'cpu'] if device == 'auto' else [device]

    for backend in backends:
        try:
            devs = jax.devices(backend)
            if not devs:
                continue
            selected = devs[0]
            jax.device_put(jnp.array(1.0, dtype=dtype), selected)
            jax.config.update('jax_default_device', selected)
            if verbose:
                print(f'psytrax: using device {selected}')
            return selected
        except Exception:
            continue

    warnings.warn(f"Could not find a compatible device for '{device}', using default JAX device.")
    return jax.devices()[0]


def available_devices():
    """Print all devices JAX can see."""
    for d in jax.devices():
        print(d)
