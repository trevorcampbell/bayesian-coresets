import numpy as np
import ctypes
import pkgutil
import os

class GIGASearch(object):
  def __init__(self, data):
    if not data.flags['C_CONTIGUOUS']:
      raise ValueError('GIGASearchC: data must be c_contiguous')
    if not data.ndim == 2:
      raise ValueError('GIGASearchC: data must be 2d')

    hcfn = pkgutil.get_loader('hilbertcoresets').filename
    self.libct = ctypes.cdll.LoadLibrary(os.path.join(hcfn, 'libgigasearch.so'))
 
    #spawns a thread to build a new tree and returns immediately
    self.libct.GIGASearch_new.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_uint, ctypes.c_uint]
    self.libct.GIGASearch_new.restype = ctypes.c_void_p
    #spawns a thread to build a new tree and returns immediately
    self.libct.GIGASearch_del.argtypes = [ctypes.c_void_p]
    self.libct.GIGASearch_del.restype = None
    #check whether the tree is done building
    self.libct.GIGASearch_check_build.argtypes = [ctypes.c_void_p]
    self.libct.GIGASearch_check_build.restype = ctypes.c_bool
    #cancel the build process
    self.libct.GIGASearch_cancel_build.argtypes = [ctypes.c_void_p]
    self.libct.GIGASearch_cancel_build.restype = None
    #get the number of build O(<,>) ops
    self.libct.GIGASearch_num_build_ops.argtypes = [ctypes.c_void_p]
    self.libct.GIGASearch_num_build_ops.restype = ctypes.c_double
    #get the number of search O(<,>) ops
    self.libct.GIGASearch_num_search_ops.argtypes = [ctypes.c_void_p]
    self.libct.GIGASearch_num_search_ops.restype = ctypes.c_double
    #get the number of searched nodes
    self.libct.GIGASearch_num_search_nodes.argtypes = [ctypes.c_void_p]
    self.libct.GIGASearch_num_search_nodes.restype = ctypes.c_double
    #perform a search (if tree not done building yet, waits on it)
    self.libct.GIGASearch_search.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    self.libct.GIGASearch_search.restype = ctypes.c_int

    self.ptr = self.libct.GIGASearch_new(data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), data.shape[0], data.shape[1])

  def num_search_ops(self):
    return self.libct.GIGASearch_num_search_ops(self.ptr)

  def num_search_nodes(self):
    return self.libct.GIGASearch_num_search_nodes(self.ptr)
  
  def num_build_ops(self):
    return self.libct.GIGASearch_num_build_ops(self.ptr)

  def is_build_done(self):
    return self.libct.GIGASearch_check_build(self.ptr)

  def cancel_build(self):
    self.libct.GIGASearch_cancel_build(self.ptr)

  def search(self, yw, y_yw):
    return self.libct.GIGASearch_search(self.ptr, yw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), y_yw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

  def __del__(self):
    self.libct.GIGASearch_del(self.ptr)

