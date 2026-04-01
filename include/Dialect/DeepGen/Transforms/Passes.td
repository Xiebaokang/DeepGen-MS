#ifndef DEEPGEN_PASSES
#define DEEPGEN_PASSES

include "mlir/Pass/PassBase.td"

// interface pass
// def AsukaSimplify: InterfacePass<"asuka-simplify", "FunctionOpInterface"> {
//   let summary = "Asuka simplify";
//   let description = [{
//     remove redundent op
//   }];

//   let constructor = "mlir::asuka::createSimplifyPass()";

//   let dependentDialects = [
//     "mlir::asuka::AsukaDialect",
//   ];
// }

def DeepGenPipeline: Pass<"DeepGen-Pipline", "::mlir::DeepGen::ForOp"> {
  let summary = "DeepGen ";
  let description = [{
    pipline
  }];

  let constructor = "mlir::DeepGen::createPipelinePass()";

  let dependentDialects = [
    "mlir::DeepGen::DeepGenDialect",
  ];
}

#endif // DEEPGEN_PASSES