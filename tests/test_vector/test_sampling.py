import bayesiancoresets as bc
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(100)
tol = 1e-9

n_trials = 10
anms = ['IS', 'RND']
algs = [bc.VectorSamplingCoreset, bc.VectorUniformSamplingCoreset]
algs_nms = zip(anms, algs)
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
#-coreset size <= M at iteration M, weights are nonnegative
#-error() vs output y(weights) are close to each other
#-reset() resets the alg properly
####################################################
def coreset_single(N, D, dist, algn):
  x = gendata(N, D, dist)
  xsave = x.copy()
  xs = x.sum(axis=0)
  anm, alg = algn
  coreset = alg(x, use_cached_xw=True)
  accuratecoreset = alg(x, use_cached_xw=False)

  #incremental M tests
  for m in range(1, N+1):
    coreset.build(m)
    accuratecoreset.build(m)
    
    #check if coreset is valid
    assert (coreset.weights() > 0.).sum() <= m, anm+" failed: coreset size > m"
    assert (coreset.weights() > 0.).sum() == coreset.size(), anm+" failed: sum of coreset.weights()>0  not equal to size(): sum = " + str((coreset.weights()>0).sum()) + " size(): " + str(coreset.size())
    assert np.all(coreset.weights() >= 0.), anm+" failed: coreset has negative weights"

    xw = (coreset.weights()[:, np.newaxis]*x).sum(axis=0)
    xwopt = (coreset.weights(optimal_scaling=True)[:, np.newaxis]*x).sum(axis=0)
  
    #check if x was modified
    assert np.fabs(x-xsave).sum() < tol, anm + " failed: modified external data directly"
 
    #check if coreset is computing error properly
    #without optimal scaling
    assert np.fabs(coreset.error() - np.sqrt(((xw-xs)**2).sum())) < tol, anm+" failed: x(w) est is not close to true x(w): est err = " + str(coreset.error()) + ' true err = ' + str(np.sqrt(((xw-xs)**2).sum()))
    #with optimal scaling (only if x(w) norm is not small / optimal scaling unstable)
    if np.sqrt((xw**2).sum()) > tol:
      assert np.fabs(coreset.error(optimal_scaling=True) - np.sqrt(((xwopt-xs)**2).sum())) < tol, anm+" failed: x(w) est is not close to true x(w) with optimal scaling: est err = " + str(coreset.error(optimal_scaling=True)) + ' true err = ' + str(np.sqrt(((xwopt-xs)**2).sum()))

    #check if fast / accurate error estimates are close
    #without optimal scaling
    assert np.fabs(coreset.error() - accuratecoreset.error()) < tol*1000, anm+" failed: error(accurate/fast) do not return similar results: fast err = " + str(coreset.error()) + ' acc err = ' + str(accuratecoreset.error())
    #with optimal scaling 
    if np.sqrt((xw**2).sum()) > tol:
      assert np.fabs(coreset.error(optimal_scaling=True) - coreset.error(optimal_scaling=True)) < tol*1000, anm+" failed: error(accurate/fast) with optimal scaling do not return similar results: fast err = " + str(coreset.error(optimal_scaling=True)) + ' acc err = ' + str(coreset.error(optimal_scaling=True))

    #ensure optimally scaled error is lower than  regular
    if np.sqrt((xw**2).sum()) > tol:
      assert coreset.error(optimal_scaling=True) - coreset.error() < tol, anm+" failed: optimal scaled coreset produces higher acc error than regular one. Optimal err = " + str(coreset.error(optimal_scaling=True)) + ' regular err: ' + str(coreset.error())

  #check reset
  coreset.reset()
  assert coreset.M == 0 and np.all(np.fabs(coreset.weights()) == 0.) and np.fabs(coreset.error() - np.sqrt((xs**2).sum())) < tol and not coreset.reached_numeric_limit, anm+" failed: reset() did not properly reset"

  
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

