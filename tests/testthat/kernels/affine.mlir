module @jit__lambda attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<5x3xf64>, %arg1: tensor<3xf64>, %arg2: tensor<f64>) -> (tensor<5xf64> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<5x3xf64>, tensor<3xf64>) -> tensor<5xf64>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<5xf64>
    %2 = stablehlo.add %0, %1 : tensor<5xf64>
    return %2 : tensor<5xf64>
  }
}
