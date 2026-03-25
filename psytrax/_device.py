"""JAX device selection utilities.

GPU installation:
  Apple Silicon (Metal):  pip install jax-metal
  NVIDIA CUDA 12:         pip install jax[cuda12]
  NVIDIA CUDA 11:         pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

After installing the appropriate backend, pass device='gpu' (or 'auto') to psytrax.fit().
"""

import warnings
import jax


def setup_device(device='auto', verbose=True):
    """Select a JAX compute device.

    Args:
        device : 'auto' | 'cpu' | 'gpu' | 'tpu'
            'auto' tries GPU/TPU first and falls back to CPU.
        verbose : bool, print the selected device.

    Returns:
        The JAX device object that was selected.
    """
    backends = ['gpu', 'tpu', 'cpu'] if device == 'auto' else [device]

    for backend in backends:
        try:
            devs = jax.devices(backend)
            if devs:
                selected = devs[0]
                jax.config.update('jax_default_device', selected)
                if verbose:
                    print(f'psytrax: using device {selected}')
                return selected
        except RuntimeError:
            continue

    # Should never reach here (cpu always exists), but just in case
    warnings.warn(f"Could not find device '{device}', using default JAX device.")
    return jax.devices()[0]


def available_devices():
    """Print all devices JAX can see."""
    for d in jax.devices():
        print(d)
