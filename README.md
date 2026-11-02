# xlaport

Ship kernels written in JAX inside an R package, and call them as ordinary R
functions with no Python at runtime.

Python exports each kernel once as StableHLO, XLA's portable program format. On
the user's machine, [`pjrt`](https://github.com/r-xla/pjrt) compiles it for
whatever device is there (CPU, any NVIDIA GPU) and runs it.

```
JAX function ── xlaport_export.export() ──▶ inst/xla/NAME.mlir + manifest.json
                                                     │  (ship with your R package)
R user ── xlaport::xla_kernel("NAME", dir) ──▶ compile on first call (pjrt/XLA) ──▶ run
```

Minimal by design: one R file, one Python file. See "Not yet" for what is
missing.

## Install

```r
install.packages("pjrt", repos = c("https://r-xla.r-universe.dev", getOption("repos")))
pjrt::install_pjrt()            # CPU plugin, plus CUDA if an NVIDIA GPU is found
# then xlaport from this repository, e.g. remotes::install_github("btraven00/xlaport")  (private for now: needs GITHUB_PAT)
```

## Shipping a kernel: three steps

### 1. Export it (Python, once per release)

Install the exporter, from a local checkout or from git with a pinned tag; it is
not on PyPI yet:

```
pip install ./xlaport/python
pip install "git+https://github.com/btraven00/xlaport@<tag>#subdirectory=python"
```

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)        # needed for float64 kernels
from xlaport_export import export

def affine(X, b, c):
    return X @ b + c

S = jax.ShapeDtypeStruct
export(affine, [S((100, 3), jnp.float64), S((3,), jnp.float64), S((), jnp.float64)],
       name="affine", out="mypkg/inst/xla", arg_names=["X", "b", "c"], out_names=["y"])
```

Run it with `JAX_PLATFORMS=cpu`. This writes `inst/xla/affine.mlir` and adds an
entry to `inst/xla/manifest.json`: the file, the input and output names,
shapes and dtypes, and the JAX version. Export more kernels into the same
directory and they share one manifest.

### 2. Ship it in your R package

Add `xlaport` to `Imports`, keep `inst/xla/` in the package, and wrap the
kernel:

```r
affine <- function(X, b, c) {
  k <- xlaport::xla_kernel("affine", system.file("xla", package = "mypkg"))
  k(X, b, c)
}
```

### 3. Users call it

```r
mypkg::affine(X, b, 2)
```

The first call compiles the program for the current device; later calls in the
same session reuse it. The device comes from the `PJRT_PLATFORM` environment
variable ("cpu" by default, "cuda" for NVIDIA). `xla_kernels(dir)` returns
every kernel in a directory as a named list, and printing a kernel shows its
signature:

```
<xla_kernel nb_fit>
  in:  Y f64[200,1000], X f64[1000,4], offset f64[1000], phi f64[200]
  out: coef f64[200,4], converged pred[200]
```

A kernel with one output returns that array; with several, it returns a named
list. Rank-1 results come back as plain R vectors.

## Rules for portable kernels

- **Fixed shapes.** A program is exported for exact input shapes, and `xla_kernel`
  refuses anything else with the exported signature in the error. For data of
  varying size, export a few size buckets and pad: zero rows, or zero
  observation weights if your model has them.
- **Flat inputs and outputs.** Arrays only, no nested structures (pytrees);
  fields that are `None` are dropped from outputs. Scalars are shape `[]`.
- **No `custom_call`s.** `jnp.linalg.*` and some other operations lower to
  LAPACK/cuSOLVER calls that tie the program to one platform. The exporter
  refuses them unless `allow_custom_calls=True`. Small solves can be written
  out (unrolled Cholesky) instead.
- **float64 needs `jax_enable_x64`** at export time. Consumer and workstation
  GPUs run f64 at 1/64 of the f32 rate.
- **Matmul precision.** The exporter lowers at "highest" precision by default,
  because JAX's default makes f32 matmuls TF32 on NVIDIA GPUs (about 1e-3
  relative error). Pass `matmul_precision="default"` to opt out.
- **Compile time.** Large kernels can take tens of seconds to over a minute to
  compile on a GPU, once per session. Tell your users.

## Checked

- `tests/testthat`: a single-output f64 kernel and a two-output f32 kernel match
  base R; wrong shapes and argument counts give clear errors. The test kernels
  are rebuilt with `tests/testthat/make_kernels.py`.
- A real kernel: the glm-jax negative-binomial fit (200 genes × 1,000 cells, 4
  covariates) exported this way matches Python to 1.8e-15 from R on CPU. The
  first call took 0.41 s with the compile, later calls 0.011 s.

## Not yet

- **A compile cache across sessions.** `pjrt` 0.6.0 has no way to save a
  compiled executable. That needs a small `pjrt` addition (XLA supports
  serialising executables).
- **Size buckets and padding hooks**, so one kernel serves any data size.
- **Versioned artifacts.** StableHLO text carries no compatibility guarantee
  across XLA versions; JAX's versioned `jax.export` format does. Whether `pjrt`
  loads it is untested.
- **GPU from R.** Untested here so far; only CPU is verified.

Background and roadmap: `~/phd/glm-jax/docs/r-package-plan.md`.
