set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})


set(LLVM_TARGET_DEFINITIONS DeepGenOps.td)
mlir_tablegen(DeepGenOps.h.inc -gen-op-decls)
mlir_tablegen(DeepGenOps.cpp.inc -gen-op-defs)
add_mlir_doc(DeepGenOps DeepGenOps DeepGen/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS DeepGenDialect.td)
mlir_tablegen(DeepGenDialect.h.inc -gen-dialect-decls)
mlir_tablegen(DeepGenDialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(DeepGenDialect DeepGenDialect DeepGen/ -gen-dialect-doc)

add_public_tablegen_target(DeepGenTableGen)