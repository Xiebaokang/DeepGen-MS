#include "ffi.h"

PYBIND11_MODULE(DeepGen_ffi, m) {
  m.doc() = "TODO";
  mlir::DeepGen::init_ffi_ir(m.def_submodule("ir"));
  mlir::DeepGen::init_ffi_passes(m.def_submodule("passes"));
}