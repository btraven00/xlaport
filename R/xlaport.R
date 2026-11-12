#' Run exported XLA programs (StableHLO) as R functions
#'
#' A kernel directory holds StableHLO text files plus a `manifest.json` that
#' names each kernel's file, inputs and outputs (written by
#' `python/xlaport_export.py`). `xla_kernel()` turns one entry into an R
#' function; `xla_kernels()` does all of them. Programs compile on first call,
#' on the user's machine, for the device in use (`PJRT_PLATFORM`, default
#' "cpu"), and stay compiled for the rest of the session.
#'
#' @param name Kernel name (a key of the manifest's `kernels`).
#' @param dir Kernel directory, e.g. `system.file("xla", package = "yourpkg")`.
#' @param device Passed to [pjrt::pjrt_compile()] and [pjrt::pjrt_buffer()];
#'   `NULL` uses the default device of `PJRT_PLATFORM`.
#' @return A function taking the kernel's inputs in manifest order. It returns
#'   one array, or a named list when the kernel has several outputs.
#' @export
xla_kernel <- function(name, dir, device = NULL) {
  k <- xla_manifest(dir)$kernels[[name]]
  if (is.null(k)) stop(sprintf("no kernel '%s' in %s", name, file.path(dir, "manifest.json")))
  make_kernel(name, k, dir, device)
}

make_kernel <- function(name, k, dir, device) {
  file <- normalizePath(file.path(dir, k$file), mustWork = TRUE)
  ins <- k$inputs
  out_names <- vapply(k$outputs, function(o) o$name, "")
  force(device)
  f <- function(...) {
    args <- list(...)
    if (length(args) != length(ins)) {
      stop(sprintf("%s takes %d inputs (%s), got %d", name, length(ins),
                   paste(vapply(ins, function(i) i$name, ""), collapse = ", "), length(args)))
    }
    bufs <- Map(to_buffer, args, ins, MoreArgs = list(device = device, kernel = name))
    res <- do.call(pjrt::pjrt_execute, c(list(compiled(file, device)), unname(bufs)))
    if (!is.list(res)) res <- list(res)
    res <- lapply(res, function(b) {
      a <- pjrt::as_array(b)
      if (length(dim(a)) == 1) dim(a) <- NULL  # rank-1 results as plain R vectors (as.vector would strip integer64)
      a
    })
    if (length(res) == 1) res[[1]] else stats::setNames(res, out_names)
  }
  structure(f, class = c("xla_kernel", "function"), kernel = name, spec = k)
}

#' @rdname xla_kernel
#' @return `xla_kernels()`: a named list of kernel functions.
#' @export
xla_kernels <- function(dir, device = NULL) {
  ks <- xla_manifest(dir)$kernels  # read once, not once per kernel
  stats::setNames(lapply(names(ks), function(n) make_kernel(n, ks[[n]], dir, device)), names(ks))
}

#' @export
print.xla_kernel <- function(x, ...) {
  k <- attr(x, "spec")
  sig <- function(v) paste(vapply(v, function(a) sprintf("%s %s[%s]", a$name, a$dtype,
                                                         paste(unlist(a$shape), collapse = ",")), ""),
                           collapse = ", ")
  cat(sprintf("<xla_kernel %s>\n  in:  %s\n  out: %s\n", attr(x, "kernel"), sig(k$inputs), sig(k$outputs)))
  invisible(x)
}

xla_manifest <- function(dir) {
  path <- file.path(dir, "manifest.json")
  if (!file.exists(path)) stop("no manifest.json in ", dir)
  m <- jsonlite::read_json(path, simplifyVector = FALSE)
  if (!identical(m$xlaport, 1L)) stop("unsupported manifest version in ", path)
  m
}

# compiled executables for this session, keyed by program file and device
.cache <- new.env(parent = emptyenv())

compiled <- function(file, device) {
  key <- paste(file, if (is.null(device)) Sys.getenv("PJRT_PLATFORM", "cpu") else format(device))
  if (is.null(.cache[[key]])) {
    prog <- pjrt::pjrt_program(path = file, format = "mlir")
    .cache[[key]] <- if (is.null(device)) pjrt::pjrt_compile(prog) else pjrt::pjrt_compile(prog, device = device)
  }
  .cache[[key]]
}

to_buffer <- function(x, spec, device, kernel) {
  want <- as.integer(unlist(spec$shape))
  have <- if (is.null(dim(x))) (if (length(want) == 0 && length(x) == 1) integer(0) else length(x)) else dim(x)
  if (!identical(as.integer(have), want)) {
    stop(sprintf("%s: input '%s' has shape [%s], the kernel was exported for [%s]", kernel, spec$name,
                 paste(have, collapse = ","), paste(want, collapse = ",")))
  }
  dev <- if (is.null(device)) list() else list(device = device)
  if (length(want) == 0) do.call(pjrt::pjrt_scalar, c(list(x, dtype = spec$dtype), dev))
  else do.call(pjrt::pjrt_buffer, c(list(x, dtype = spec$dtype), dev))
}
