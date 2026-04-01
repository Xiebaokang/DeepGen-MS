#ifndef DEEPGENTODEEPGENGPU_CONVERSION_PASSES
#define DEEPGENTODEEPGENGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertDeepGenToDeepGenGPU: Pass<"convert-DeepGen-to-DeepGenGPU", "::mlir::ModuleOp"> {
  let summary = "convert DeepGen to DeepGen GPU";
  let description = [{}];

  let constructor = "mlir::DeepGen::createConvertDeepGenToDeepGenGPUPass()";

  let dependentDialects = [
    "mlir::DeepGen::DeepGenDialect"
  ];
}



#endif // DEEPGENTODEEPGENGPU_CONVERSION_PASSES