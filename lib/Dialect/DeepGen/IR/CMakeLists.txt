add_DeepGen_library(DeepGenIR
  DeepGenOps.cpp

  DEPENDS
  DeepGenTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)