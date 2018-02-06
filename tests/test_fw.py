import bayesiancoresets as bc
import numpy as np
import warnings


np.random.seed(1)

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them

n_trials = 10
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
#-error() vs output y(weights) are close to each other
#-error is decreasing
#-reset() resets the alg properly
#-run(M) with increasing M outputs same weights as 
# one run with large M
#-if input size = 1, error is 0 for any M
#-if input is colinear, error is 0 forall M
#-sqrt_bound()/exp_bound is valid
#-bound is positive, decreasing -> 0 
#-fw outputs expected weights and error for axis_aligned data 
####################################################
def fw_single(N, D, dist="gauss"):
  x = gendata(N, D, dist)
  xs = x.sum(axis=0)
  fw = bc.FrankWolfe(x)

  #bound tests
  prev_sqrt_bd = np.inf
  prev_exp_bd = np.inf
  for m in range(1, N+1):
    sqrt_bd = fw.sqrt_bound(m)
    exp_bd = fw.exp_bound(m)
    assert sqrt_bd >= 0., "FW failed: sqrt bound < 0 " + str(sqrt_bd)
    assert sqrt_bd - prev_sqrt_bd < tol, "FW failed: sqrt bound is not decreasing"
    assert exp_bd >= 0., "FW failed: exp bound < 0"
    assert exp_bd - prev_exp_bd < tol, "FW failed: exp bound is not decreasing"
    prev_sqrt_bd = sqrt_bd
    prev_exp_bd = exp_bd
  assert fw.sqrt_bound(1e100) < tol, "FW failed: sqrt bound doesn't approach 0"
  assert fw.exp_bound(1e100) < tol, "FW failed: exp bound doesn't approach 0"

  #incremental M tests
  prev_err = np.inf
  for m in range(1, N+1):
    fw.run(m)
    if x.shape[0] == 1:
      assert np.fabs(fw.weights() - np.array([1])) < tol or (np.fabs(fw.weights() - np.array([0])) < tol and (x**2).sum() == 0.), "FW failed: coreset not immediately optimal with N = 1"
    assert (fw.weights() > 0.).sum() <= m, "FW failed: coreset size > m"
    xw = (fw.weights()[:, np.newaxis]*x).sum(axis=0)
    assert np.sqrt(((xw-xs)**2).sum()) - prev_err < tol, "FW failed: error is not monotone decreasing"
    assert np.fabs(fw.error('accurate') - np.sqrt(((xw-xs)**2).sum())) < tol, "FW failed: x(w) est is not close to true x(w)"
    assert np.fabs(fw.error('accurate') - fw.error()) < tol*1000, "FW failed: fw.error(accurate/fast) do not return similar results"
    assert fw.sqrt_bound() - np.sqrt(((xw-xs)**2).sum()) >= -tol, "FW failed: sqrt bound invalid"
    assert fw.exp_bound() - np.sqrt(((xw-xs)**2).sum()) >= -tol, "FW failed: exp bound invalid"
    if 'colinear' in dist and m >= 2:
      assert np.sqrt(((xw-xs)**2).sum()) < tol, "FW failed: for M>=2, coreset with colinear data not optimal"
    if 'axis' in dist:
      assert np.all( np.fabs(fw.weights()[ fw.weights() > 0. ] - float(N)/float(m) ) < tol ), "FW failed: on axis-aligned data, weights are not N/M"
      assert np.fabs(np.sqrt(((xw-xs)**2).sum())/np.sqrt((xs**2).sum()) - np.sqrt(float(N)/float(m) - 1.)) < tol, "FW failed: on axis-aligned data, error is not sqrt(N/M-1)"
    prev_err = np.sqrt(((xw-xs)**2).sum())
  #save incremental M result
  w_inc = fw.weights()
  xw_inc = (fw.weights()[:, np.newaxis]*x).sum(axis=0) 
  
  #check reset
  fw.reset()
  assert fw.M == 0 and np.all(np.fabs(fw.weights()) < tol) and np.fabs(fw.error() - np.sqrt((xs**2).sum())) < tol and not fw.reached_numeric_limit, "FW failed: fw.reset() did not properly reset"
  #check run up to N all at once vs incremental
  fw.run(N)
  xw = (fw.weights()[:, np.newaxis]*x).sum(axis=0) 
  assert np.sqrt(((xw-xw_inc)**2).sum()) < tol, "FW failed: incremental run up to N doesn't produce same result as one run at N"

def test_fw():
  for N, D, dist in tests:
    for n in range(n_trials):
      yield fw_single, N, D, dist
 
####################################################
#verifies that FW correctly responds to bad input
####################################################
    
def test_fw_input_validation():
  try:
    bc.FrankWolfe('fdas')
  except ValueError:
    pass
  except:
    assert False, "Unrecognized error type"
  try:
    bc.FrankWolfe(np.array(['fdsa', 'asdf']))
  except ValueError:
    pass
  except:
    assert False, "Unrecognized error type"
   

