#ifndef ASUKA_CONVERSION_DEEPGENTODEEPGENGPU_DEEPGENTODEEPGENGPU_PASS_H
#define ASUKA_CONVERSION_DEEPGENTODEEPGENGPU_DEEPGENTODEEPGENGPU_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir::DeepGen {

std::unique_ptr<mlir::Pass> createConvertDeepGenToDeepGenGPUPass();

#define GEN_PASS_REGISTRATION
#include "Conversion/DeepGenToDeepGenGPU/Passes.h.inc"

} // namespace mlir::DeepGen

#endif