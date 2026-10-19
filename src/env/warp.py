import os

# CUDA graphs mjx's warp FFI keeps per callable (its default: 32)
WARP_GRAPH_CACHE = 4


def configure_warp() -> None:
    require_warp_memory()
    # a CUDA graph (GBs of scratch) per buffer-address set; our separate jitted
    # calls pass new addresses, so the default cache runs out of memory;
    # the setting has no public import path, only mjx's vendored warp
    from mujoco.mjx.third_party.warp._src.jax_experimental import ffi

    ffi.set_jax_callable_default_graph_cache_max(WARP_GRAPH_CACHE)


def require_warp_memory() -> None:
    # warp allocates collision scratch outside JAX's pool on every step;
    # JAX's default 75% preallocation leaves it too little
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise RuntimeError(
            "sim.backend=warp needs XLA_PYTHON_CLIENT_PREALLOCATE=false in the "
            "environment before the process starts (the justfile and the "
            "slurm/ scripts export it); otherwise warp runs out of GPU memory "
            "mid-run. Or run on sim.backend=jax"
        )
