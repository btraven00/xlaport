# xlaport

Call JAX functions from R, with no Python at runtime.

You export a JAX function once as StableHLO, XLA's portable program format, and
ship the file inside your R package. On the user's machine,
[`pjrt`](https://github.com/r-xla/pjrt) compiles it for their device (CPU or
NVIDIA GPU) and `xlaport` wraps it as an ordinary R function.

## Install

```r
install.packages("pjrt", repos = c("https://r-xla.r-universe.dev", getOption("repos")))
remotes::install_github("btraven00/xlaport")
pjrt::install_pjrt()   # once: downloads the XLA plugin
```

Package authors also need the Python exporter:

```
pip install "git+https://github.com/btraven00/xlaport#subdirectory=python"
```

## Use

**1. Export the function** (Python, once per release):

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)        # needed for float64
from xlaport_export import export

def affine(X, b, c):
    return X @ b + c

S = jax.ShapeDtypeStruct
export(affine, [S((100, 3), jnp.float64), S((3,), jnp.float64), S((), jnp.float64)],
       name="affine", out="mypkg/inst/xla", arg_names=["X", "b", "c"], out_names=["y"])
```

Run it with `JAX_PLATFORMS=cpu`. It writes `inst/xla/affine.mlir` and records
the kernel's inputs and outputs in `inst/xla/manifest.json`. Several kernels
can share one directory.

**2. Wrap it in your R package** (add `xlaport` to `Imports`):

```r
affine <- function(X, b, c) {
  k <- xlaport::xla_kernel("affine", system.file("xla", package = "mypkg"))
  k(X, b, c)
}
```

**3. Users call it** like any R function: `mypkg::affine(X, b, 2)`.

The first call compiles the program; later calls in the same session reuse it.
The device is set by the `PJRT_PLATFORM` environment variable (`cpu` by
default, `cuda` for NVIDIA).

`xla_kernels(dir)` returns every kernel in a directory as a named list.
Printing a kernel shows its signature:

```
<xla_kernel affine>
  in:  X f64[100,3], b f64[3], c f64[]
  out: y f64[100]
```

A kernel with one output returns that array; with several, a named list.
Rank-1 outputs come back as plain vectors, and 64-bit integers as `integer64`.

## What a kernel must look like

- **Fixed shapes.** A program is exported for exact input shapes, and anything
  else is refused with the expected signature in the error. For data of varying
  size, export a few sizes and pad.
- **Arrays in, arrays out.** No nested structures. Supported types are
  float32, float64, int32, int64 and bool.
- **No `custom_call`.** `jnp.linalg.*` and some other operations compile to
  LAPACK or cuSOLVER calls that only work on one platform. The exporter refuses
  them unless you pass `allow_custom_calls=True`.
- **float32 matmuls are exported at full precision.** JAX's default would run
  them as TF32 on NVIDIA GPUs (about 1e-3 relative error). Pass
  `matmul_precision="default"` to opt out.

## Limits

- **Compiling happens every session.** `pjrt` 0.7.0 cannot save a compiled
  program. Small kernels compile in under a second on CPU; large ones can take
  a minute or more on a GPU.
- **No built-in padding or size buckets.** [`glmjax`](https://github.com/btraven00/glm-jax/tree/main/r)
  shows one way to do it on top of `xla_kernels()`.
- **Programs are tied to the XLA version.** StableHLO text has no compatibility
  guarantee across versions, so re-export and re-test when `pjrt` updates.
- **Only CPU is tested from R.**
