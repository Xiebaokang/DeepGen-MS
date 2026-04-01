add_DeepGen_library(DeepGenToDeepGenGPU
  DeepGenToDeepGenGPUPass.cpp

  DEPENDS
  DeepGenConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  DeepGenIR
)