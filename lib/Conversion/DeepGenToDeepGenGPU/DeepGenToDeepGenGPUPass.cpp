#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialect/DeepGen/IR/DeepGenDialect.h"

namespace mlir::DeepGen {

#define GEN_PASS_DEF_CONVERTDEEPGENTODEEPGENGPU
#include "Conversion/DeepGenToDeepGenGPU/Passes.h.inc"

} // namespace mlir::DeepGen

namespace mlir::DeepGen {

namespace {

struct KernelOpConversion : public OpConversionPattern<KernelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(KernelOp kernelOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    FunctionType funcType = mlir::dyn_cast<FunctionType>(kernelOp.getFunctionType());
    ArrayRef<Type> inputTypes = funcType.getInputs();
    // new func
    func::FuncOp funcOp = rewriter.create<func::FuncOp>(kernelOp.getLoc(), kernelOp.getSymName(), funcType);
    auto& region = funcOp->getRegion(0);
    region.emplaceBlock();
    auto& body = funcOp.front();
    SmallVector<Location> locs(inputTypes.size(), kernelOp.getLoc());
    body.addArguments(inputTypes, locs);

    auto& oldBlock = kernelOp->getRegion(0).front();
    auto& newBlock = funcOp->getRegion(0).front();
    // replace all uses with
    for (unsigned i=0; i<oldBlock.getNumArguments(); ++i) {
        Value oldArg = oldBlock.getArgument(i);
        Value newArg = newBlock.getArgument(i);
        oldArg.replaceAllUsesWith(newArg);
    }
    // move operation from origin kernelOp
    newBlock.getOperations().splice(newBlock.getOperations().begin(), oldBlock.getOperations());
    // llvm::outs() <<  << "\n";
    rewriter.eraseOp(&(newBlock.back()));
    // add returnop
    rewriter.setInsertionPointToEnd(&body);
    rewriter.create<func::ReturnOp>(funcOp.getLoc());
    // remove origin kernelOp
    rewriter.eraseOp(kernelOp);
    return success();
  }
};

struct ParallelOpConversion : public OpConversionPattern<ParallelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ParallelOp parallelOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    constexpr gpu::Dimension dims[] = {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
    SmallVector<Value, 4> bids;
    // create gpu blockidx
    auto grid = parallelOp.getGrid();
    rewriter.setInsertionPoint(parallelOp);
    for (unsigned i=0; i<grid.size(); i++) {
      auto bidOp = rewriter.create<gpu::BlockIdOp>(parallelOp.getLoc(), dims[i]);
      bidOp->setAttr("range", rewriter.getI32IntegerAttr(grid[i]));
      bids.push_back(bidOp);
    }
    // create gpu threadIdx
    auto tidOp = rewriter.create<gpu::ThreadIdOp>(parallelOp.getLoc(), dims[0]);
    tidOp->setAttr("range", rewriter.getI32IntegerAttr(parallelOp.getThreadNum()));
    // collect
    auto& block = parallelOp->getRegion(0).front();
    SmallVector<Operation*> opsToMove;
    for (auto &op : block.getOperations()) {
      if (!op.hasTrait<OpTrait::IsTerminator>()) {
        opsToMove.push_back(&op);
      }
    }
    // move
    Operation *pos = parallelOp.getOperation();
    for (Operation *op : opsToMove) {
        op->moveAfter(pos);
        pos = op;
    }
    // replace uses
    for (unsigned i=0; i<block.getNumArguments(); ++i) {
      Value oldArg = block.getArgument(i);
      oldArg.replaceAllUsesWith(bids[i]);
    }
    rewriter.eraseOp(parallelOp);
    return success();
  }
};

struct ForOpConversion : public OpConversionPattern<ForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ForOp dforOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    uint64_t lb = dforOp.getLower();
    uint64_t ub = dforOp.getUpper();
    uint64_t step = dforOp.getStep();
    Value div = dforOp.getInductionVar();
    Value aiv;
    auto aforOp = rewriter.create<affine::AffineForOp>(dforOp.getLoc(), lb, ub, step, mlir::ValueRange({}), 
      [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
        aiv = iv;
      });
    // move
    aforOp.getBody()->getOperations().splice(aforOp.getBody()->getOperations().begin(), dforOp.getBody()->getOperations());
    rewriter.eraseOp(&(aforOp.getBody()->back()));
    rewriter.setInsertionPointToEnd(aforOp.getBody());
    rewriter.create<affine::AffineYieldOp>(aforOp.getLoc());
    // replace
    div.replaceAllUsesWith(aiv);
    rewriter.eraseOp(dforOp);
    return success();
  }
};

class ConvertDeepGenToDeepGenGPU : public ::mlir::DeepGen::impl::ConvertDeepGenToDeepGenGPUBase<ConvertDeepGenToDeepGenGPU> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    ConversionTarget target(*context);

    target.addLegalDialect<mlir::arith::ArithDialect,
                           mlir::affine::AffineDialect,
                           mlir::memref::MemRefDialect,
                           mlir::gpu::GPUDialect,
                           mlir::func::FuncDialect, 
                           mlir::math::MathDialect>();
    // target.addIllegalDialect<mlir::DeepGen::DeepGenDialect>();
    target.addIllegalOp<KernelOp>();
    target.addIllegalOp<ParallelOp>();
    target.addIllegalOp<ForOp>();
    RewritePatternSet patterns(context);
    patterns.add<KernelOpConversion>(context);
    patterns.add<ParallelOpConversion>(context);
    patterns.add<ForOpConversion>(context);
    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertDeepGenToDeepGenGPUPass() {
  return std::make_unique<ConvertDeepGenToDeepGenGPU>();
}

} // namespace mlir::DeepGen