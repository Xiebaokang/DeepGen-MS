from ..DeepGen_ffi import ir
from ..DeepGen_ffi import passes
import os
from enum import Enum

class DType(str, Enum):
  E4M3FN = "e4m3fn"
  E5M2 = "e5m2"
  FLOAT16 = "float16"
  FLOAT32 = "float32"
  FLOAT64 = "float64"

def get_dtype(b, d):
  if d == DType.E4M3FN:
    return b.get_e4m3fn_ty()
  elif d == DType.E5M2:
    return b.get_e5m2_ty()
  elif d == DType.FLOAT16:
    return b.get_f16_ty()
  elif d == DType.FLOAT32:
    return b.get_f32_ty()
  elif d == DType.FLOAT64:
    return b.get_f64_ty()

class classproperty(object):
  def __init__(self, f):
    self.f = f
    
  def __get__(self, obj, owner):
    return self.f(owner)

# memory space
class MemorySpace:
  _backend = None # 'cuda' or 'rocm'
  @classmethod
  def get_backend(cls):
    if cls._backend is None:
      if os.environ.get("USE_ROCM") == "1":
        cls._backend = "rocm"
      else:
        cls._backend = "cuda"
    return cls._backend
  
  @classproperty
  def LOCAL(cls):
    return 5 if cls.get_backend() == "rocm" else 0
  
  @classproperty
  def GLOBAL(cls):
    return 1
  
  @classproperty
  def SHARED(cls):
    return 3
    


# pass mamager
def get_pass_manager(op, context=None):
  if isinstance(op, ir.module):
    assert context is None
    context = op.context
    top_pm = passes.pass_manager(context)
    pm = top_pm
  else:
    assert context is not None
    top_pm = passes.pass_manager(context)
    pm = top_pm
  return top_pm, pm


def create_module():
  context = ir.context()
  ir.DeepGen.load_dialects(context)
  builder = ir.builder(context)
  module = builder.create_module()
  module.context = context
  return module

def create_kernel(module, name, **kwargs):
  dtype_count, shape_count = 0, 0
  shapes, dtypes = [None]*len(kwargs), [None]*len(kwargs)
  for key, word in kwargs.items():
    if "shape" in key:
      if not isinstance(word, tuple):
        raise TypeError("shape must be of type tuple")
      shapes[int(key[5:])] = word
      shape_count += 1
    elif "dtype" in key:
      if not isinstance(word, DType):
        raise TypeError("shape must be of type DType")
      if not key[4:].isdigit():
        dtypes[0] = word
      else:
        dtypes[int(key[4:])] = word
      dtype_count += 1

  if not (shape_count != dtype_count and dtype_count == 1):
    raise ValueError("create_kernel: Invalid kwargs (dtype and shape)")
  # create kernelOp
  context = module.context
  builder = ir.builder(context)
  mem_types = []
  for i in range(shape_count):
    if dtypes[i] is None:
      dty = get_dtype(builder, dtypes[0])
      mem_type = builder.get_memref_ty(dty, shapes[i])
    else:
      dty = get_dtype(builder, dtypes[i])
      mem_type = builder.get_memref_ty(dty, shapes[i])
    mem_types.append(mem_type)
  kernel_type = builder.get_kernel_ty(mem_types)
  kernel = builder.create_kernel_op(module, name, kernel_type)
  module.push_back(kernel)
  entry = kernel.add_entry_block()
  builder.set_insertion_point_to_start(entry)
  
  # create func body
  if kwargs.get("body") is None:
    raise ValueError("create_kernel: missing kwargs \"body\"")
  kernel_args = [entry.get_argument(i) for i in range(entry.get_num_arguments())]
  kwargs["body"](builder, *kernel_args)
  
  return module
  