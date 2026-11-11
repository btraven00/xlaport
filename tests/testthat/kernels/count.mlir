module @jit__lambda attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x6xf32>) -> (tensor<4xi64> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x6xf32>
    %1 = stablehlo.compare GT, %arg0, %0, FLOAT : (tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<4x6xi1>
    %2 = stablehlo.convert %1 : (tensor<4x6xi1>) -> tensor<4x6xi32>
    %3 = stablehlo.convert %2 : (tensor<4x6xi32>) -> tensor<4x6xi64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.add across dimensions = [1] : (tensor<4x6xi64>, tensor<i64>) -> tensor<4xi64>
    return %4 : tensor<4xi64>
  }
}
