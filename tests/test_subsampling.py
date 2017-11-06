import hilbertcoresets as hc
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them


n_trials = 20
tol = 1e-9
tests = [(N, D, dist) for N in [1, 10, 100] for D in [1, 3, 10] for dist in ['gauss', 'bin', 'gauss_colinear', 'bin_colinear', 'axis_aligned']]

def gendata(N, D, dist="gauss"):
  if dist == "gauss":
    x = np.random.normal(0., 1., (N, D))
  elif dist == "bin":
    x = (np.random.rand(N, D) > 0.5).astype(float)
  elif dist == "gauss_colinear":
    x = np.random.normal(0., 1., D)
    y = np.random.rand(N)*2.-1.
    x = y[:, np.newaxis]*x
  elif dist == "bin_colinear":
    x = (np.random.rand(D) > 0.5).astype(float)
    y = np.random.rand(N)*2.-1.
    x = y[:, np.newaxis]*x
  else:
    x = np.zeros((N, N))
    for i in range(N):
      x[i, i] = 1./float(N)
  return x

####################################################
#verifies that 
#-coreset size <= M at iteration M
#-bound is positive and decreasing -> 0
#-error() vs output y(weights) are close to each other
#-reset() resets the alg properly
####################################################
def is_single(N, D, dist="gauss", use_rnd=False):
  x = gendata(N, D, dist)
  xs = x.sum(axis=0)
  if use_rnd:
    IS = hc.RandomSubsampling(x)
  else:
    IS = hc.ImportanceSampling(x)

  #TODO remove if statement once randomsubsampling bound implemented
  if not use_rnd:
    #bound tests
    delta = 0.05
    prev_bd = np.inf
    for m in range(1, N+1):
      bd = IS.sqrt_bound(delta, m)
      assert bd >= 0., "IS failed: sqrt bound < 0, " + str(bd)
      assert bd - prev_bd < tol, "IS failed: sqrt bound is not decreasing"
      prev_bd = bd
    assert IS.sqrt_bound(delta, 1e100) < tol, "IS failed: sqrt bound doesn't approach 0"
  
  for m in range(1, N+1):
    IS.run(m)
    assert (IS.weights()>0).sum() <= m, "IS failed: coreset size > m"
    xw = (IS.weights()[:, np.newaxis]*x).sum(axis=0)
    assert np.fabs(IS.error() - np.sqrt(((xw-xs)**2).sum())) < tol, "IS failed: x(w) est is not close to true x(w)"

  IS.reset()
  assert IS.M == 0 and np.all(np.fabs(IS.weights()) < tol) and np.fabs(IS.error() - np.sqrt((xs**2).sum())) < tol, "IS failed: IS.reset() did not properly reset"

  #run for max int number of iterations (fast since it just uses np.random.multinomial)
  IS.run(np.iinfo(np.int64).max)
  xw = (IS.weights()[:, np.newaxis]*x).sum(axis=0)
  assert IS.error() < 1e-5 and np.sqrt(((xw-xs)**2).sum()) < 1e-5, "IS failed: IS did not converge to optimum after int64.max iterations."

def test_is():
  for N, D, dist in tests:
    for n in range(n_trials):
      yield is_single, N, D, dist, False

def test_rs():
  for N, D, dist in tests:
    for n in range(n_trials):
      yield is_single, N, D, dist, True
 
def test_is_hinv():
  x = np.random.rand(10, 3)
  alg = hc.ImportanceSampling(x)
  y = np.random.rand(10000)*10000
  h = lambda y : (1.+y)*np.log(1.+y)-y
  for i in range(10000):
    assert np.fabs(h(alg._hinv(y[i])) - y[i]) < tol*y[i] and np.fabs(alg._hinv(h(y[i])) - y[i]) < tol*y[i], "IS failed: h is not inv of _hinv"

####################################################
#verifies that IS correctly responds to bad input
####################################################
   
def test_is_input_validation():
  try:
    hc.ImportanceSampling('fdas')
  except ValueError:
    pass
  except:
    assert False, "Unrecognized error type"
  try:
    hc.ImportanceSampling(np.array(['fdsa', 'asdf']))
  except ValueError:
    pass
  except:
    assert False, "Unrecognized error type"
 
  
 

