from bayesiancoresets.base import FullDataCoreset
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(100)
tol = 1e-9

def test_full_data():
  for N in [0, 1, 10]:
    coreset = FullDataCoreset(N)
    for m in [1, 10, 100]:
      coreset.build(m)
      assert coreset.error() < tol, "FullDataCoreset failed: error not 0"
      assert np.all(coreset.weights() == np.ones(N)), "FullDataCoreset failed: weights not ones"
      assert coreset.M == N, "FullDataCoreset failed: coreset.M not equal to N after build"
    #check reset
    coreset.reset()


def test_initialization():
  for N in [0, 1, 10]:
    coreset = FullDataCoreset(N)
    assert coreset.N == N, "FullDataCoreset failed: N was not set properly"
    try:
      coreset.error()
    except NotImplementedError:
      pass
    except:
      assert False, "FullDataCoreset failed: Allowed error() before build"
    assert coreset.initialized, "FullDataCoreset failed: did not initialize"
    assert coreset.wts.shape[0] == coreset.N, "FullDataCoreset failed: probabilities do not sum to 1: sum = " + str(coreset.ps.sum())

def test_reset():
  for N in [0, 1, 10]:
    coreset = FullDataCoreset(N)
    for m in [1, 10, 100]:
      coreset.build(m)
      #check reset
      coreset.reset()
      assert coreset.M == 0 and np.all(np.fabs(coreset.weights()) == 0.) and not coreset.reached_numeric_limit, "FullDataset failed: reset() did not properly reset"

def test_build():
  for N in [0, 1, 10]:
    coreset = FullDataCoreset(N)
    for m in [1, 10, 100]:
      coreset.build(m)
      assert coreset.error() < tol, "FullDataCoreset failed: error not 0"
      assert np.all(coreset.weights() == np.ones(N)), "FullDataCoreset failed: weights not ones"
      assert coreset.M == N, "FullDataCoreset failed: M should always be = m"
      assert np.all(coreset.wts >= 0), "FullDataCoreset failed: weights must be nonnegative"


