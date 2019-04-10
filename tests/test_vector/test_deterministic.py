import bayesiancoresets as bc
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(100)
tol = 1e-9


n_trials = 10
anms = ['GIGA', 'FW', 'MP', 'FSW', 'OMP', 'LAR', 'LASSO']
algs = [bc.GIGACoreset, bc.FrankWolfeCoreset, bc.MatchingPursuitCoreset, bc.ForwardStagewiseCoreset, bc.OrthoPursuitCoreset, bc.LARCoreset, bc.LassoCoreset]
algs_nms = list(zip(anms, algs))
tests = [(N, D, dist, algn) for N in [1, 10, 100] for D in [1, 3, 10] for dist in ['gauss', 'bin', 'gauss_colinear', 'bin_colinear', 'axis_aligned'] for algn in algs_nms]


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
#-build(M) with increasing M outputs same weights as one run with large M
#-if input size = 1, error is 0 for any M
#-if input is colinear, error is 0 forall M
####################################################
def coreset_single(N, D, dist, algn):
  x = gendata(N, D, dist)
  print(x)
  xs = x.sum(axis=0)
  anm, alg = algn
  coreset = alg(x, use_cached_xw=True)
  accuratecoreset = alg(x, use_cached_xw=False)

  #incremental M tests
  prev_err = np.inf
  for m in range(1, N+1):
    coreset.build(m)
    accuratecoreset.build(m)
    #check if coreset for 1 datapoint is immediately optimal
    if x.shape[0] == 1:
      assert np.fabs(coreset.weights(optimal_scaling=True) - np.array([1])) < tol or (np.fabs(coreset.weights(optimal_scaling=True) - np.array([0])) < tol and (x**2).sum() == 0.), anm +" failed: coreset not immediately optimal with N = 1. weights: " + str(coreset.weights(optimal_scaling=True))
    #check if coreset is valid
    assert (coreset.weights() > 0.).sum() <= m, anm+" failed: coreset size > m"
    assert (coreset.weights() > 0.).sum() == coreset.size(), anm+" failed: sum of coreset.weights()>0  not equal to size(): sum = " + str((coreset.weights()>0).sum()) + " size(): " + str(coreset.size())
    assert np.all(coreset.weights() >= 0.), anm+" failed: coreset has negative weights"
    
    xw = (coreset.weights()[:, np.newaxis]*x).sum(axis=0)
    xwopt = (coreset.weights(optimal_scaling=True)[:, np.newaxis]*x).sum(axis=0)
 
    #check if actual output error is monotone
    assert np.sqrt(((xw-xs)**2).sum()) - prev_err < tol, anm+" failed: error is not monotone decreasing, err = " + str(np.sqrt(((xw-xs)**2).sum())) + " prev_err = " +str(prev_err) + " M = " + str(coreset.M) + " N = " + str(N) + " m = " + str(m)   + " coreset.err = " + str(coreset.error()) + " xwopt = " + str(xwopt) + " w = " + str(coreset.wts) + " wext = " + str(coreset.weights()) + " xw = " + str(xw) + " coreset xw = " + str(coreset.xw) + " xs = " + str(xs)  + " prevwts = " + str(prev_wts) + " data = " + str(coreset.x)

#+ ' M_cache: '+ str( coreset.M_cache if anm == 'LASSO' else 0) + ' w_cache: '+ str( coreset.w_cache if anm == 'LASSO' else 0) + ' lmb_cache: '+ str( coreset.lmb_cache if anm == 'LASSO' else 0)
    #check if coreset is computing error properly
    #without optimal scaling
    assert np.fabs(coreset.error() - np.sqrt(((xw-xs)**2).sum())) < tol, anm+" failed: x(w) est is not close to true x(w): est err = " + str(coreset.error()) + ' true err = ' + str(np.sqrt(((xw-xs)**2).sum()))
    #with optimal scaling
    assert np.fabs(coreset.error(optimal_scaling=True) - np.sqrt(((xwopt-xs)**2).sum())) < tol, anm+" failed: x(w) est is not close to true x(w) with optimal scaling: est err = " + str(coreset.error(optimal_scaling=True)) + ' true err = ' + str(np.sqrt(((xwopt-xs)**2).sum()))

    #check if fast / accurate error estimates are close
    #without optimal scaling
    assert np.fabs(coreset.error() - accuratecoreset.error()) < tol*1000, anm+" failed: error(accurate/fast) do not return similar results: fast err = " + str(coreset.error()) + ' acc err = ' + str(accuratecoreset.error())
    #with optimal scaling
    assert np.fabs(accuratecoreset.error(optimal_scaling=True) - coreset.error(optimal_scaling=True)) < tol*1000, anm+" failed: error(accurate/fast) with optimal scaling do not return similar results: fast err = " + str(coreset.error(optimal_scaling=True)) + ' acc err = ' + str(coreset.error(optimal_scaling=True))

    #ensure optimally scaled error is lower than  regular
    assert coreset.error(optimal_scaling=True) - coreset.error() < tol, anm+" failed: optimal scaled coreset produces higher acc error than regular one. Optimal err = " + str(coreset.error(optimal_scaling=True)) + ' regular err: ' + str(coreset.error())

    #if data are colinear, check if the coreset is optimal immediately
    if 'colinear' in dist and m >= 1:
      assert np.sqrt(((xwopt-xs)**2).sum()) < tol, anm + " failed: colinear data, m>= 1 not immediately optimal:  optimal scaled err " + str(np.sqrt(((xwopt-xs)**2).sum())) + " tol = " + str(tol) + " m = " + str(m)
    ##if data are axis aligned, 
    #if 'axis' in dist:
    #  assert np.all( np.fabs(coreset.weights()[ coreset.weights() > 0. ] - 1. ) < tol ), anm+" failed: on axis-aligned data, weights are not 1"
    #  assert np.fabs(np.sqrt(((xw-xs)**2).sum())/np.sqrt((xs**2).sum()) - np.sqrt(1. - float(m)/float(N))) < tol, anm+" failed: on axis-aligned data, error is not sqrt(1 - M/N)"
    prev_err = np.sqrt(((xw-xs)**2).sum())
    prev_wts = coreset.wts.copy()
  #save incremental M result
  w_inc = coreset.weights()
  xw_inc = (coreset.weights()[:, np.newaxis]*x).sum(axis=0) 

  #check reset
  coreset.reset()
  assert coreset.M == 0 and np.all(np.fabs(coreset.weights()) == 0.) and np.fabs(coreset.error() - np.sqrt((xs**2).sum())) < tol and not coreset.reached_numeric_limit, anm+" failed: reset() did not properly reset"

  #check build up to N all at once vs incremental
  #do this test for all except bin, where symmetries can cause instabilities in the choice of vector / weights
  if dist != 'bin':
    coreset.build(N)
    xw = (coreset.weights()[:, np.newaxis]*x).sum(axis=0) 
    assert np.sqrt(((xw-xw_inc)**2).sum()) < tol, anm+" failed: incremental buid up to N doesn't produce same result as one run at N : \n xw = " + str(xw) + " error = " +str(np.sqrt(((xw-xs)**2).sum())) + " \n xw_inc = " + str(xw_inc) + " error = " +  str(np.sqrt(((xw_inc-xs)**2).sum())) + " \n xs = " +str(xs)


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
  except:
    fe1 = True
  try:
    alg(np.array(['fdsa', 'asdf']))
  except:
    fe2 = True

  if not fe1 or not fe2:
    assert False, anm + " failed: did not catch invalid input"

