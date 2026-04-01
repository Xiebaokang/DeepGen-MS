import DeepGen as dp

def matmul(M, N, K, BM, BN, BK, thread_num, dtype=dp.DType.FLOAT32):
  shapeA, shapeB, shapeC =(K, M), (K, N), (M, N)
  grid = (M // BM, N // BN)

  def gemm_kernel_body(builder, *args):
    A, B, C = args
    context = builder.get_context()
    #create parallelOp
    prarllel = builder.create_parallel_op(grid, thread_num)
    entry2 = prarllel.add_entry_block()
    builder.set_insertion_point_to_start(entry2)
    shared_A = builder.create_alloc_buffer_op((BK, BM), dp.DType.FLOAT32, dp.MemorySpace.SHARED, 128).get_result()
    shared_B = builder.create_alloc_buffer_op((BK, BN), dp.DType.FLOAT32, dp.MemorySpace.SHARED, 128).get_result()
    fragment_C = builder.create_alloc_buffer_op((BM, BN), dp.DType.FLOAT32, dp.MemorySpace.GLOBAL).get_result()
    # init fragment_c
    fill = builder.create_fill_op(fragment_C, 0.0, dp.DType.FLOAT32)
    # block_idx
    bx, by = prarllel.get_ivs()
    d0 = builder.get_affine_dim_expr(0)
    d1 = builder.get_affine_dim_expr(1)
    map1 = dp.AffineMap.get(2, 0, (d0, d1 * BM), context)
    map2 = dp.AffineMap.get(2, 0, (d0, d1 * BN), context)
    map3 = dp.AffineMap.get(2, 0, (d0 * BM, d1 * BN), context)
    map4 = dp.AffineMap.get_constants_map([0, 0], context)
    # body 
    def for_body_test(b, iv):
      cp1 = builder.create_copy_op(A, shared_A, (iv, by), (), map1, map4)
      cp2 = builder.create_copy_op(B, shared_B, (iv, bx), (), map2, map4)
      # create gemmOp
      gemm = b.create_gemm_op(shared_A, shared_B, fragment_C, True, False)
    # create forOp
    builder.create_for_op(0, K, BK, for_body_test, 1)
    # store
    cp3 = builder.create_copy_op(fragment_C, C, [], [by, bx], map4, map3)
  
  # create kernel
  mod = dp.create_module()
  mod = dp.create_kernel(mod, "Gemm_Kernel", 
                         shape0=shapeA, shape1=shapeB, shape2=shapeC, 
                         dtype=dtype, body=gemm_kernel_body)
  mod.dump()
  # run pass
  top_pm, pm = dp.get_pass_manager(mod)
  dp.add_DeepGen_to_DeepGenGPU(pm)
  top_pm.run(mod)
  mod.dump()

if __name__ == "__main__":
  auto_configs = {
    "BM": 128, "BN": 128, "BK": 32
  }
  matmul(1024, 1024, 1024, 128, 128, 32, 128)