#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/DeepGen/IR/DeepGenDialect.h"
#include "Dialect/DeepGen/Transforms/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"

namespace mlir::DeepGen {

// MLIR 的 Pass 生成机制要求在包含 .inc 文件之前，必须定义一个特定的宏来启用该 Pass 的基类定义。
#define GEN_PASS_DEF_DEEPGENPIPELINE
#include "Dialect/DeepGen/Transforms/Passes.h.inc"

} // namespace mlir::DeepGen

namespace mlir::DeepGen {
namespace {

class PipelinePass : public ::mlir::DeepGen::impl::DeepGenPipelineBase<PipelinePass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ForOp forOp = getOperation();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createPipelinePass() { return std::make_unique<PipelinePass>(); }

} // namespace mlir::DeepGen