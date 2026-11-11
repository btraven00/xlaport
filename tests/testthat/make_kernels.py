"""Regenerate the test kernels: JAX_PLATFORMS=cpu python tests/testthat/make_kernels.py"""
import os, sys
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from xlaport_export import export

out = os.path.join(os.path.dirname(__file__), "kernels")
S = jax.ShapeDtypeStruct
export(lambda X, b, c: X @ b + c, [S((5, 3), jnp.float64), S((3,), jnp.float64), S((), jnp.float64)],
       name="affine", out=out, arg_names=["X", "b", "c"], out_names=["y"])
export(lambda A: (A.mean(axis=1), jax.nn.logsumexp(A, axis=1)), [S((4, 6), jnp.float32)],
       name="rowstats", out=out, arg_names=["A"], out_names=["mean", "logsumexp"])
export(lambda A: jnp.sum(A > 0, axis=1), [S((4, 6), jnp.float32)], name="count", out=out, arg_names=["A"], out_names=["n"])
