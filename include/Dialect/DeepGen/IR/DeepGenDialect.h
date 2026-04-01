#ifndef DEEPGEN_DIELACT_H_
#define DEEPGEN_DIELACT_H_

// dialect
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"


#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect/DeepGen/IR/DeepGenDialect.h.inc"

#include "Dialect/DeepGen/Utils/Utils.h"

// #include "Dialect/DeepGen/IR/DeepGenEnums.h.inc"

#define GET_OP_CLASSES
#include "Dialect/DeepGen/IR/DeepGenOps.h.inc"


#endif