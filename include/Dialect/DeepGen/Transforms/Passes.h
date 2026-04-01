#ifndef DEEPGEN_TRANSFORMS_PASSES_H_
#define DEEPGEN_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir::DeepGen {

// FIXME: remove all manual constructors in tablegen and use the default one?
#define GEN_PASS_DECL
#include "Dialect/DeepGen/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createPipelinePass();

#define GEN_PASS_REGISTRATION
#include "Dialect/DeepGen/Transforms/Passes.h.inc"

} // namespace mlir::DeepGen

#endif