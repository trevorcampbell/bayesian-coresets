import hilbertcoresets as hc
import numpy as np

n_trials = 50
tol = 1e-9

####################################################
#verifies that 
#-coreset size <= M at iteration M
#-bound is positive and decreasing -> 0
#-
#-error() vs output y(weights) are close to each other
#-error is decreasing
#-reset() resets the alg properly
#-run(M) with increasing M outputs same weights as 
# one run with large M
#-if input size = 1, error is 0 for any M
#-if input is colinear, error is 0 forall M
#-sqrt_bound()/exp_bound is valid
#-fw outputs expected weights and error for axis_aligned data 
####################################################
def is_single(N, D, dist="gauss"):
  print 'IS single: N = ' +str(N) + ' D = ' +str(D) + ' dist = ' + dist
  for n in range(n_trials):
    if dist == "gauss":
      x = np.random.normal(0., 1., (N, D))
    elif dist == "bin":
      x = (np.random.rand(N, D) > 0.5).astype(int)
    elif dist == "gauss_colinear":
      x = np.random.normal(0., 1., D)
      y = np.random.rand(N)*2.-1.
      x = y[:, np.newaxis]*x
    elif dist == "bin_colinear":
      x = (np.random.rand(D) > 0.5).astype(int)
      y = np.random.rand(N)*2.-1.
      x = y[:, np.newaxis]*x
    else:
      x = np.zeros((N, N))
      for i in range(N):
        x[i, i] = 1./float(N)

    xs = x.sum(axis=0)
    IS = hc.ImportanceSampling(x)

    #bound tests
    delta = 0.05
    prev_bd = np.inf
    for m in range(1, N+1):
      bd = IS.sqrt_bound(delta, m)
      assert bd >= 0., "IS failed: sqrt bound < 0"
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
    assert IS.error() < tol and np.sqrt(((xw-xs)**2).sum()) < tol, "IS failed: IS did not converge to optimum after int64.max iterations"


def test_fw_random():
  tests = [(N, D, dist) for N in [0, 1, 1000] for D in [0, 1, 10] for dist in ['gauss', 'bin']]
  for N, D, dist in tests:
    yield is_single(N, D, dist)
 
def test_fw_colinear():
  tests = [(N, D, dist) for N in [0, 1, 1000] for D in [0, 1, 10] for dist in ['gauss_colinear', 'bin_colinear']]
  for N, D, dist in tests:
    yield is_single(N, D, dist)

def test_fw_axis_aligned():
  for N in [0, 1, 10, 1000]:
    yield is_single(N, 0, 'axis_aligned')

def test_is_hinv():
  x = np.random.rand(10, 3)
  alg = hc.ImportanceSampling(x)
  y = np.random.rand(10000)*10000
  h = lambda y : (1.+y)*np.log(1.+y)-y
  for i in range(10000):
    assert np.fabs(h(alg._hinv(y)) - y) < tol and np.fabs(alg._hinv(h(y)) - y) < tol, "IS failed: h is not inv of _hinv"


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
 
  
 

