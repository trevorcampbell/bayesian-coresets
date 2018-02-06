import numpy as np
from bayesiancoresets import gigasearch as gs

np.random.seed(1)

n_trials = 50
tol = 1e-9
tests = [(N, D, dist) for N in [10, 100] for D in [3, 10] for dist in ['gauss', 'bin', 'gauss_colinear', 'bin_colinear', 'axis_aligned']]


def gendata(N, D, dist="gauss"):
  if dist == "gauss":
    x = np.random.normal(0., 1., (N, D))
  elif dist == "bin":
    x = np.zeros((N, D))
    nz = (x**2).sum(axis=1) == 0.
    while nz.sum() > 0:
      x[nz, :] = (np.random.rand(nz.sum(), D) > 0.5).astype(float)
      nz = (x**2).sum(axis=1) == 0.
  elif dist == "gauss_colinear":
    x = np.random.normal(0., 1., D)
    y = np.random.rand(N)*2.-1.
    x = y[:, np.newaxis]*x
  elif dist == "bin_colinear":
    x = (np.random.rand(D) > 0.5).astype(float)
    while (x**2).sum() == 0:
      x = (np.random.rand(D) > 0.5).astype(float)
    y = np.random.rand(N)*2.-1.
    x = y[:, np.newaxis]*x
  else:
    x = np.zeros((N, N))
    for i in range(N):
      x[i, i] = 1./float(N)
  return x/np.sqrt((x**2).sum(axis=1))[:, np.newaxis]


def gigasearch_single(N, D, dist="gauss"):
  x = gendata(N, D, dist)
  srch = gs.GIGASearch(x)
  for m in range(n_trials):
    yw = np.random.normal(0., 1., x.shape[1])
    yw /= np.sqrt((yw**2).sum())
    y_yw = np.random.normal(0., 1., x.shape[1])
    y_yw -= y_yw.dot(yw)*yw
    y_yw /= np.sqrt((y_yw**2).sum())
    n_ot = srch.search(yw, y_yw)
    f_ot = x[n_ot, :].dot(y_yw)/np.sqrt(1.-x[n_ot, :].dot(yw)**2)
    n_ol = (x.dot(y_yw)/np.sqrt(1.-x.dot(yw)**2)).argmax()
    f_ol = x[n_ol, :].dot(y_yw)/np.sqrt(1.-x[n_ol, :].dot(yw)**2)
    assert f_ol - f_ot < tol, "gigasearch failed; true obj = " + str(f_ol) + " gigasearch obj = " + str(f_ot)
      
def test_gigasearch():
  for N, D, dist in tests:
    for n in range(n_trials):
      yield gigasearch_single, N, D, dist


def test_gigasearch_stability():
  X = np.random.rand(10, 5)

  #test stability with immediate garbage collection
  for i in range(100):
    gs.GIGASearch(X)
  
  #test stability with cancellation
  for i in range(100):
    tr = gs.GIGASearch(X)
    tr.cancel_build()
  
  #test stability with search then cancellation
  for i in range(100):
    yw = np.random.normal(0., 1., 5)
    yw /= np.sqrt((yw**2).sum())
    y_yw = np.random.normal(0., 1., 5)
    y_yw -= y_yw.dot(yw)*yw
    y_yw /= np.sqrt((y_yw**2).sum())
    tr = gs.GIGASearch(X)
    tr.search(yw, y_yw)
    tr.cancel_build()
  
  #test stability with cancellation then search
  for i in range(100):
    yw = np.random.normal(0., 1., 5)
    yw /= np.sqrt((yw**2).sum())
    y_yw = np.random.normal(0., 1., 5)
    y_yw -= y_yw.dot(yw)*yw
    y_yw /= np.sqrt((y_yw**2).sum())
    tr = gs.GIGASearch(X)
    tr.cancel_build()
    tr.search(yw, y_yw)
  
  #test stability with lots of searches
  tr = gs.GIGASearch(X)
  for i in range(100):
    yw = np.random.normal(0., 1., 5)
    yw /= np.sqrt((yw**2).sum())
    y_yw = np.random.normal(0., 1., 5)
    y_yw -= y_yw.dot(yw)*yw
    y_yw /= np.sqrt((y_yw**2).sum())
    tr.search(yw, y_yw)

if __name__ == '__main__':
  import time
  import sys
  import ctypes
  import pkgutil
  import os
  N = 10000000
  D = 20
  x = gendata(N, D, 'gauss')

  hcfn = pkgutil.get_loader('hilbertcoresets').filename
  libgs = ctypes.cdll.LoadLibrary(os.path.join(hcfn, 'libgigasearch.so'))
  libgs.search.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_uint, ctypes.c_uint]
  libgs.search.restype = ctypes.c_int
  for i in range(1000):
    yw = np.random.normal(0., 1., D)
    yw /= np.sqrt((yw**2).sum())
    y_yw = np.random.normal(0., 1., D)
    y_yw -= y_yw.dot(yw)*yw
    y_yw /= np.sqrt((y_yw**2).sum())
    t0 = time.time()
    num = (x*y_yw).sum(axis=1)
    denom = (x*yw).sum(axis=1)
    denom = np.sqrt(1.-denom**2)
    n_ol = (num/denom).argmax()
    lin_time = time.time()-t0
    lin_n = N
    t0 = time.time() 
    n_og = libgs.search(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), yw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), y_yw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), x.shape[0], x.shape[1])
    #n_og = giga.search(yw, y_yw)
    gs_time = time.time()-t0
    gs_n = N
    sys.stderr.write(' gs_t: ' + str(gs_time)  + ' lin_t: ' + str(lin_time) + ' gs_n: ' + str(n_og) + ' lin_n: ' + str(n_ol) + ' gs_n: ' + str(gs_n)  + ' lin_n: ' + str(lin_n)+ '\n')
    sys.stderr.flush()
    
    
    





