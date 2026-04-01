#ifndef DEEPGEN_DIALECT
#define DEEPGEN_DIALECT

include "mlir/IR/OpBase.td"


def DeepGen_Dialect : Dialect {
  let name = "dp";

  let cppNamespace = "::mlir::DeepGen";

  let summary = "DeepGen Dialact in MLIR";

  let description = [{
    DeepGen Dialect.
  }];

  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::cf::ControlFlowDialect",
    "::mlir::gpu::GPUDialect"
  ];

  // let useDefaultAttributePrinterParser = 1;
}

#endif // DEEPGEN_DIALECT