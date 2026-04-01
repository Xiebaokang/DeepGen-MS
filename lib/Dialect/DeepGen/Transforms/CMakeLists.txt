add_DeepGen_library(DeepGenTransforms
  pipeline.cpp

  DEPENDS
  DeepGenTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  DeepGenIR
)