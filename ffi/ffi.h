#ifndef DEEPGEN_FFI_H_
#define DEEPGEN_FFI_H_

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace mlir::DeepGen {

void init_ffi_ir(py::module_ &&m);

void init_ffi_passes(py::module_ &&m);

} // namespace mlir::DeepGen

#endif // DEEPGEN_FFI_H_