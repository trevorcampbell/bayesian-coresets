import bayesiancoresets as bc
import numpy as np
import warnings

#anms = ['GIGA', 'FW', 'RP', 'FSW', 'OMP', 'LAR', 'IS', 'RND']
#algs = [bc.GIGA, bc.FrankWolfe, bc.ReweightedPursuit, bc.ForwardStagewise, bc.OrthoPursuit, bc.LAR, bc.ImportanceSampling, bc.RandomSubsampling]


np.random.seed(1)

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them

n_trials = 10
tol = 1e-6
anms = ['IS', 'RND']
algs = [bc.ImportanceSampling, bc.RandomSubsampling]
algs_nms = zip(anms, algs)
tests = [(N, D, dist, algn) for N in [1, 10, 100] for D in [1, 3, 10] for dist in ['gauss', 'bin', 'gauss_colinear', 'bin_colinear', 'axis_aligned'] for algn in algs_nms]


tests = [(N, D, dist, algn) for N in [10] for D in [3] for dist in ['gauss'] for algn in algs_nms]

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
#-run(M) with increasing M outputs same weights as one run with large M
#-if input size = 1, error is 0 for any M
#-if input is colinear, error is 0 forall M
#-sqrt_bound()/exp_bound is valid
#-bound is positive, decreasing -> 0 
#-giga outputs expected weights and error for axis_aligned data 
####################################################
def coreset_single(N, D, dist, algn):
  x = gendata(N, D, dist)
  xs = x.sum(axis=0)
  anm, alg = algn
  coreset = alg(x)

  #incremental M tests
  prev_err = np.inf
  for m in range(1, N+1):
    coreset.run(m)
    if x.shape[0] == 1:
      assert np.fabs(coreset.weights() - np.array([1])) < tol or (np.fabs(coreset.weights() - np.array([0])) < tol and (x**2).sum() == 0.), anm +" failed: coreset not immediately optimal with N = 1"
    assert (coreset.weights() > 0.).sum() <= m, anm+" failed: coreset size > m"
    xw = (coreset.weights()[:, np.newaxis]*x).sum(axis=0)
    assert np.sqrt(((xw-xs)**2).sum()) - prev_err < tol, anm+" failed: error is not monotone decreasing, err = " + str(np.sqrt(((xw-xs)**2).sum())) + " prev_err = " +str(prev_err) + " M = " + str(coreset.M)
    assert np.fabs(coreset.error('accurate') - np.sqrt(((xw-xs)**2).sum())) < tol, anm+" failed: x(w) est is not close to true x(w): est err = " + str(coreset.error('accurate')) + ' true err = ' + str(np.sqrt(((xw-xs)**2).sum()))
    assert np.fabs(coreset.error('accurate') - coreset.error()) < tol*1000, anm+" failed: error(accurate/fast) do not return similar results: fast err = " + str(coreset.error()) + ' acc err = ' + str(coreset.error('accurate'))
    if 'colinear' in dist and m >= 1:
      if not np.sqrt(((xw-xs)**2).sum()) < tol:
        assert False, "colinear m>= 1 problem nrm = " +str(np.sqrt(((xw-xs)**2).sum())) + " tol = " + str(tol) + " m = " + str(m)
      assert np.sqrt(((xw-xs)**2).sum()) < tol, anm+" failed: for M>=2, coreset with colinear data not optimal"
    if 'axis' in dist:
      assert np.all( np.fabs(coreset.weights()[ coreset.weights() > 0. ] - 1. ) < tol ), anm+" failed: on axis-aligned data, weights are not 1"
      assert np.fabs(np.sqrt(((xw-xs)**2).sum())/np.sqrt((xs**2).sum()) - np.sqrt(1. - float(m)/float(N))) < tol, anm+" failed: on axis-aligned data, error is not sqrt(1 - M/N)"
    prev_err = np.sqrt(((xw-xs)**2).sum())
  #save incremental M result
  w_inc = coreset.weights()
  xw_inc = (coreset.weights()[:, np.newaxis]*x).sum(axis=0) 

  #check reset
  coreset.reset()
  assert coreset.M == 0 and np.all(np.fabs(coreset.weights()) == 0.) and np.fabs(coreset.error() - np.sqrt((xs**2).sum())) < tol and not coreset.reached_numeric_limit, anm+" failed: reset() did not properly reset"
  #check run up to N all at once vs incremental
  #do this test for all except bin, where symmetries can cause instabilities in the choice of vector (and then different weights if the original vector norms were different)
  if dist != 'bin':
    coreset.run(N)
    xw = (coreset.weights()[:, np.newaxis]*x).sum(axis=0) 
    assert np.sqrt(((xw-xw_inc)**2).sum()) < tol, anm+" failed: incremental run up to N doesn't produce same result as one run at N : \n xw = " + str(xw) + " error = " +str(np.sqrt(((xw-xs)**2).sum())) + " \n xw_inc = " + str(xw_inc) + " error = " +  str(np.sqrt(((xw_inc-xs)**2).sum())) + " \n xs = " +str(xs)

def test_coreset():
  for N, D, dist, alg in tests:
    for n in range(n_trials):
      yield coreset_single, N, D, dist, alg

####################################################
#verifies that cst construction correctly responds to bad input
####################################################
    
def test_coreset_input_validation():
  for anm, alg in algs_nms:
    yield input_validation_single, alg, anm 

def input_validation_single(alg, anm):
  fe1 = False
  fe2 = False
  try:
    alg('fdas')
  except ValueError:
    fe1 = True
    pass
  except:
    assert False, anm + " failed: produced unrecognized error type"
  try:
    alg(np.array(['fdsa', 'asdf']))
  except ValueError:
    fe2 = True
    pass
  except:
    assert False, anm + " failed: produced unrecognized error type"

  if not fe1 or not fe2:
    assert False, anm + " failed: did not catch invalid input"

