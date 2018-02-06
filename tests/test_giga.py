import bayesiancoresets as bc
import numpy as np
import warnings


np.random.seed(1)

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them

n_trials = 10
tol = 1e-6
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
#-error() vs output y(weights) are close to each other
#-error is decreasing
#-reset() resets the alg properly
#-run(M) with increasing M outputs same weights as 
# one run with large M
#-if input size = 1, error is 0 for any M
#-if input is colinear, error is 0 forall M
#-sqrt_bound()/exp_bound is valid
#-bound is positive, decreasing -> 0 
#-giga outputs expected weights and error for axis_aligned data 
####################################################
def giga_single(N, D, dist="gauss"):
  x = gendata(N, D, dist)
  xs = x.sum(axis=0)
  giga = bc.GIGA(x)

  #TODO uncomment once giga bds implemented
  ##bound tests
  #prev_sqrt_bd = np.inf
  #prev_exp_bd = np.inf
  #for m in range(1, N+1):
  #  sqrt_bd = giga.sqrt_bound(m)
  #  exp_bd = giga.exp_bound(m)
  #  assert sqrt_bd >= 0., "GIGA failed: sqrt bound < 0"
  #  assert sqrt_bd - prev_sqrt_bd < tol, "GIGA failed: sqrt bound is not decreasing"
  #  assert exp_bd >= 0., "GIGA failed: exp bound < 0"
  #  assert exp_bd - prev_exp_bd < tol, "GIGA failed: exp bound is not decreasing"
  #  prev_sqrt_bd = sqrt_bd
  #  prev_exp_bd = exp_bd
  #assert giga.sqrt_bound(1e100) < tol, "GIGA failed: sqrt bound doesn't approach 0"
  #assert giga.exp_bound(1e100) < tol, "GIGA failed: exp bound doesn't approach 0"

  #incremental M tests
  prev_err = np.inf
  for m in range(1, N+1):
    giga.run(m)
    if x.shape[0] == 1:
      assert np.fabs(giga.weights() - np.array([1])) < tol or (np.fabs(giga.weights() - np.array([0])) < tol and (x**2).sum() == 0.), "GIGA failed: coreset not immediately optimal with N = 1"
    assert (giga.weights() > 0.).sum() <= m, "GIGA failed: coreset size > m"
    xw = (giga.weights()[:, np.newaxis]*x).sum(axis=0)
    assert np.sqrt(((xw-xs)**2).sum()) - prev_err < tol, "GIGA failed: error is not monotone decreasing, err = " + str(np.sqrt(((xw-xs)**2).sum())) + " prev_err = " +str(prev_err) + " M = " + str(giga.M)
    assert np.fabs(giga.error('accurate') - np.sqrt(((xw-xs)**2).sum())) < tol, "GIGA failed: x(w) est is not close to true x(w): est err = " + str(giga.error('accurate')) + ' true err = ' + str(np.sqrt(((xw-xs)**2).sum()))
    assert np.fabs(giga.error('accurate') - giga.error()) < tol*1000, "GIGA failed: giga.error(accurate/fast) do not return similar results: fast err = " + str(giga.error()) + ' acc err = ' + str(giga.error('accurate'))
    #TODO uncomment once giga bound implemented
    #assert giga.sqrt_bound() - np.sqrt(((xw-xs)**2).sum()) >= -tol, "GIGA failed: sqrt bound invalid"
    #assert giga.exp_bound() - np.sqrt(((xw-xs)**2).sum()) >= -tol, "GIGA failed: exp bound invalid"
    if 'colinear' in dist and m >= 1:
      if not np.sqrt(((xw-xs)**2).sum()) < tol:
        assert False, "colinear m>= 1 problem nrm = " +str(np.sqrt(((xw-xs)**2).sum())) + " tol = " + str(tol) + " m = " + str(m)
      assert np.sqrt(((xw-xs)**2).sum()) < tol, "GIGA failed: for M>=2, coreset with colinear data not optimal"
    if 'axis' in dist:
      assert np.all( np.fabs(giga.weights()[ giga.weights() > 0. ] - 1. ) < tol ), "GIGA failed: on axis-aligned data, weights are not 1"
      assert np.fabs(np.sqrt(((xw-xs)**2).sum())/np.sqrt((xs**2).sum()) - np.sqrt(1. - float(m)/float(N))) < tol, "GIGA failed: on axis-aligned data, error is not sqrt(1 - M/N)"
    prev_err = np.sqrt(((xw-xs)**2).sum())
  #save incremental M result
  w_inc = giga.weights()
  xw_inc = (giga.weights()[:, np.newaxis]*x).sum(axis=0) 
  
  #check reset
  giga.reset()
  assert giga.M == 0 and np.all(np.fabs(giga.weights()) < tol) and np.fabs(giga.error() - np.sqrt((xs**2).sum())) < tol and not giga.reached_numeric_limit, "GIGA failed: giga.reset() did not properly reset"
  #check run up to N all at once vs incremental
  giga.run(N)
  xw = (giga.weights()[:, np.newaxis]*x).sum(axis=0) 
  assert np.sqrt(((xw-xw_inc)**2).sum()) < tol, "GIGA failed: incremental run up to N doesn't produce same result as one run at N : \n xw = " + str(xw) + " error = " +str(np.sqrt(((xw-xs)**2).sum())) + " \n xw_inc = " + str(xw_inc) + " error = " +  str(np.sqrt(((xw_inc-xs)**2).sum())) + " \n xs = " +str(xs)

def test_giga():
  for N, D, dist in tests:
    for n in range(n_trials):
      yield giga_single, N, D, dist

####################################################
#verifies that GIGA correctly responds to bad input
####################################################
    
def test_giga_input_validation():
  try:
    bc.GIGA('fdas')
  except ValueError:
    pass
  except:
    assert False, "Unrecognized error type"
  try:
    bc.GIGA(np.array(['fdsa', 'asdf']))
  except ValueError:
    pass
  except:
    assert False, "Unrecognized error type"

