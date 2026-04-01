#ifndef DEEPGEN_ANALYSIS_INFERLOWERINFO_H
#define DEEPGEN_ANALYSIS_INFERLOWERINFO_H

#include <variant>

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

#include "Dialect/DeepGen/IR/DeepGenDialect.h"

namespace mlir {

struct LowerInfo {
	// 缓存
  Value buffer;
  // 线程束索引映射
  AffineMap warp_index;  // [((tid / 32) / 1) * 8, ((tid / 32) % 1) * 64]
  // 线程索引映射
  AffineMap thread_index;  // [((tid % 32) / 8) * 2, ((tid % 32) % 8) * 4]
  // 线程内部索引映射
  AffineMap thread_inner_index;  // [i0, j0]
  // block离散索引
  AffineMap block_scatter_index;  // [i1 * 4 * 8, j1 * 1 * 64]
  // warp离散索引
  AffineMap warp_scatter_index;  // [i2 * 4 * 2, j2 * 8 * 4]

  // 线程范围，同时也是线程块中线程的数量
  int64_t thread_bound;  // 128
  // 线程内部索引变量取值范围，控制索引边界
  vector<int64_t> thread_inner_bound;  // [2, 4]
  //
  vector<int64_t> block_sactter_bound;  // [1, 2]
  //
  vector<int64_t> warp_scatter_bound;  // [1, 2]
  // 线程块相对于线程束的布局
  vector<int64_t> block_layout;  // [4, 1]
  // 线程束相对于线程的布局
  vector<int64_t> warp_layout;  // [4, 8]

  public:
  Value getBuffer() { return buffer; }
}


} // namespace mlir

#endif // DEEPGEN_ANALYSIS_INFERLOWERINFO_H