module @jit__lambda attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x6xf32>) -> (tensor<4xf32> {jax.result_info = "result[0]"}, tensor<4xf32> {jax.result_info = "result[1]"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x6xf32>, tensor<f32>) -> tensor<4xf32>
    %cst_0 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %2 = stablehlo.divide %0, %1 : tensor<4xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.reduce(%arg0 init: %cst_1) applies stablehlo.maximum across dimensions = [1] : (tensor<4x6xf32>, tensor<f32>) -> tensor<4xf32>
    %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %5 = stablehlo.maximum %4, %3 : tensor<4xf32>
    %6 = stablehlo.is_finite %5 : (tensor<4xf32>) -> tensor<4xi1>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %8 = stablehlo.select %6, %5, %7 : tensor<4xi1>, tensor<4xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<4xf32>) -> tensor<4x1xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x6xf32>
    %11 = stablehlo.subtract %arg0, %10 : tensor<4x6xf32>
    %12 = stablehlo.exponential %11 : tensor<4x6xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.reduce(%12 init: %cst_4) applies stablehlo.add across dimensions = [1] : (tensor<4x6xf32>, tensor<f32>) -> tensor<4xf32>
    %14 = stablehlo.abs %13 : tensor<4xf32>
    %15 = stablehlo.log %14 : tensor<4xf32>
    %16 = stablehlo.add %15, %8 : tensor<4xf32>
    return %2, %16 : tensor<4xf32>, tensor<4xf32>
  }
}
