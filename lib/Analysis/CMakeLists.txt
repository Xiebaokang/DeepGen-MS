add_DeepGen_library(DeepGenAnalysis
  InferLowerInfo.cpp

  DEPENDS
  DeepGenTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  DeepGenIR 
)