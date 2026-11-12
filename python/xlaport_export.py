"""Export jitted JAX functions as StableHLO for xlaport (R).

    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)          # for f64 kernels
    from xlaport_export import export

    def affine(X, b, c):
        return X @ b + c
    export(affine, [jax.ShapeDtypeStruct((100, 3), jnp.float64), jax.ShapeDtypeStruct((3,), jnp.float64),
                    jax.ShapeDtypeStruct((), jnp.float64)],
           name="affine", out="inst/xla", arg_names=["X", "b", "c"], out_names=["y"])

Writes OUT/NAME.mlir (StableHLO text) and adds NAME to OUT/manifest.json.
Inputs and outputs must be flat lists of arrays (no pytrees). Run with
JAX_PLATFORMS=cpu so the program is lowered without device-specific choices.
"""

import json
import os

import jax

_DTYPES = {"float64": "f64", "float32": "f32", "int64": "i64", "int32": "i32", "bool": "pred"}


def export(fn, specs, *, name, out, arg_names=None, out_names=None, allow_custom_calls=False,
           matmul_precision="highest"):
    # "highest": with JAX's default, f32 matmuls run as TF32 on NVIDIA GPUs (~1e-3 relative error)
    with jax.default_matmul_precision(matmul_precision):
        lowered = jax.jit(fn).lower(*specs)
    text = lowered.as_text()
    # custom_calls (LAPACK, cuSOLVER, ...) tie the program to one platform; refuse them by default
    if "custom_call" in text and not allow_custom_calls:
        raise ValueError(f"{name}: the program contains stablehlo.custom_call, so it is not portable "
                         "across devices (often jnp.linalg.*); rewrite it or pass allow_custom_calls=True")
    outs = jax.tree_util.tree_leaves(jax.eval_shape(fn, *specs))

    def entry(s, n):
        return {"name": n, "shape": list(s.shape), "dtype": _DTYPES[str(s.dtype)]}

    arg_names = arg_names or [f"x{i}" for i in range(len(specs))]
    out_names = out_names or [f"y{i}" for i in range(len(outs))]
    if len(arg_names) != len(specs) or len(out_names) != len(outs):
        raise ValueError(f"{name}: {len(specs)} inputs and {len(outs)} outputs, but {len(arg_names)} "
                         f"arg_names and {len(out_names)} out_names")
    bad = {str(s.dtype) for s in [*specs, *outs]} - set(_DTYPES)
    if bad:
        raise ValueError(f"{name}: unsupported dtype {sorted(bad)}; supported: {sorted(_DTYPES)}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{name}.mlir"), "w") as f:
        f.write(text)
    path = os.path.join(out, "manifest.json")
    m = json.load(open(path)) if os.path.exists(path) else {"xlaport": 1, "kernels": {}}
    m["kernels"][name] = {"file": f"{name}.mlir", "jax": jax.__version__,
                          "inputs": [entry(s, n) for s, n in zip(specs, arg_names)],
                          "outputs": [entry(s, n) for s, n in zip(outs, out_names)]}
    with open(path, "w") as f:
        json.dump(m, f, indent=1)
    return m["kernels"][name]
