#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

#include "Dialect/DeepGen/IR/DeepGenDialect.h"
#include "Dialect/DeepGen/Transforms/Passes.h"
#include "Conversion/DeepGenToDeepGenGPU/Passes.h"

#define ADD_PASS_WRAPPER_0(name, builder) m.def(name, [](mlir::OpPassManager &pm) { pm.addPass(builder()); })

#define ADD_PASS_WRAPPER_1(name, builder, ty0)                                                                         \
  m.def(name, [](mlir::OpPassManager &pm, ty0 val0) { pm.addPass(builder(val0)); })

#define ADD_PASS_WRAPPER_2(name, builder, ty0, ty1)                                                                    \
  m.def(name, [](mlir::OpPassManager &pm, ty0 val0, ty1 val1) { pm.addPass(builder(val0, val1)); })

#define ADD_PASS_WRAPPER_3(name, builder, ty0, ty1, ty2)                                                               \
  m.def(name, [](mlir::OpPassManager &pm, ty0 val0, ty1 val1, ty2 val2) { pm.addPass(builder(val0, val1, val2)); })

#define ADD_PASS_WRAPPER_4(name, builder, ty0, ty1, ty2, ty3)                                                          \
  m.def(name, [](mlir::OpPassManager &pm, ty0 val0, ty1 val1, ty2 val2, ty3 val3) {                                    \
    pm.addPass(builder(val0, val1, val2, val3));                                                                       \
  })

#define ADD_PASS_OPTION_WRAPPER_1(name, builder, ty0)                                                                  \
  m.def(name, [](mlir::OpPassManager &pm, ty0 val0) { pm.addPass(builder({val0})); })

#define ADD_PASS_OPTION_WRAPPER_2(name, builder, ty0, ty1)                                                             \
  m.def(name, [](mlir::OpPassManager &pm, ty0 val0, ty1 val1) { pm.addPass(builder({val0, val1})); })

#define ADD_PASS_OPTION_WRAPPER_3(name, builder, ty0, ty1, ty2)                                                        \
  m.def(name, [](mlir::OpPassManager &pm, ty0 val0, ty1 val1, ty2 val2) { pm.addPass(builder({val0, val1, val2})); })

#define ADD_PASS_OPTION_WRAPPER_4(name, builder, ty0, ty1, ty2, ty3)                                                   \
  m.def(name, [](mlir::OpPassManager &pm, ty0 val0, ty1 val1, ty2 val2, ty3 val3) {                                    \
    pm.addPass(builder({val0, val1, val2, val3}));                                                                     \
  })

namespace py = pybind11;
using ret = py::return_value_policy;

namespace mlir::DeepGen {

void init_common_passes(py::module &m) {
  ADD_PASS_WRAPPER_0("add_sccp", createSCCPPass);
  ADD_PASS_WRAPPER_0("add_symbol_dce", createSymbolDCEPass);
  ADD_PASS_WRAPPER_0("add_inliner", createInlinerPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer", createCanonicalizerPass);
  ADD_PASS_WRAPPER_0("add_cse", createCSEPass);
  ADD_PASS_WRAPPER_0("add_licm", createLoopInvariantCodeMotionPass);
}

void init_DeepGen_passes(py::module &m) {
  // optimize pass
  ADD_PASS_WRAPPER_0("add_DeepGen_pipeline", createPipelinePass);

  // lowering pass
  ADD_PASS_WRAPPER_0("add_DeepGen_to_DeepGenGPU", createConvertDeepGenToDeepGenGPUPass);

  // translate
  // m.def("translate_kernel_to_py", [](KernelOp kernel_op, bool import, bool benchmark) -> std::string {
  //   std::string str;
  //   llvm::raw_string_ostream os(str);
  //   if (failed(kernel_to_py_impl(kernel_op, os, import, benchmark))) {
  //     throw std::runtime_error("kernel to py failed");
  //   }
  //   return str;
  // });

  // m.def("translate_module_to_py", [](ModuleOp module_op, bool benchmark) -> std::string {
  //   std::string str;
  //   llvm::raw_string_ostream os(str);
  //   if (failed(module_to_py_impl(module_op, os, benchmark))) {
  //     throw std::runtime_error("module to py failed");
  //   }
  //   return str;
  // });
}

// TODO: impl op pass manager to utlize parallel passes
void init_ffi_passes(py::module_ &&m) {
  py::class_<OpPassManager>(m, "op_pass_manager", py::module_local());
  py::class_<PassManager, OpPassManager>(m, "pass_manager", py::module_local())
      .def(py::init<MLIRContext *>())
      .def("enable_debug",
           [](PassManager &self) {
             // TODO
           })
      .def("nest_any", &PassManager::nestAny, ret::reference)
      .def("run",
           [](PassManager &self, ModuleOp &mod) {
             // TODO: debug
             if (failed(self.run(mod.getOperation())))
               throw std::runtime_error("PassManager::run failed on module_op");
           });

  init_common_passes(m);
  init_DeepGen_passes(m);
}

} // namespace mlir::DeepGen