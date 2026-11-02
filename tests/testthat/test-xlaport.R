dir <- test_path("kernels")

test_that("single-output kernel matches R", {
  affine <- xla_kernel("affine", dir)
  X <- matrix(seq(0.5, 7.5, by = 0.5), 5, 3); b <- c(1, -2, 0.25); c0 <- 3
  expect_equal(affine(X, b, c0), drop(X %*% b + c0), tolerance = 1e-14)
  expect_equal(affine(X, b, c0), drop(X %*% b + c0), tolerance = 1e-14)  # second call: cached executable
})

test_that("multi-output f32 kernel returns a named list", {
  rowstats <- xla_kernels(dir)$rowstats
  A <- matrix(sin(1:24), 4, 6)
  r <- rowstats(A)
  expect_named(r, c("mean", "logsumexp"))
  expect_equal(r$mean, rowMeans(A), tolerance = 1e-6)
  expect_equal(r$logsumexp, log(rowSums(exp(A))), tolerance = 1e-6)
})

test_that("wrong shapes and argument counts are refused with the exported signature", {
  affine <- xla_kernel("affine", dir)
  expect_error(affine(matrix(0, 4, 3), c(1, 2, 3), 1), "has shape \\[4,3\\].*exported for \\[5,3\\]")
  expect_error(affine(matrix(0, 5, 3)), "takes 3 inputs \\(X, b, c\\), got 1")
  expect_error(xla_kernel("nope", dir), "no kernel 'nope'")
})

test_that("print shows the signature", {
  expect_output(print(xla_kernel("affine", dir)), "X f64\\[5,3\\], b f64\\[3\\], c f64\\[\\]")
})
