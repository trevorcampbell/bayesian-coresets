from bayesiancoresets.base import FullDataCoreset
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(100)
tol = 1e-9


def test_api():
  for N in [0, 1, 10]:
    coreset = FullDataCoreset(N)
    try:
      coreset.error()
    except NotImplementedError:
      pass
    except:
      assert False, "FullDataCoreset failed: Allowed error() before build"
    coreset.build(1)
    assert coreset.error() == 0., "FullDataCoreset failed: error() should be 0 after build"
    assert np.all(coreset.weights() == coreset.wts), "FullDataCoreset failed: wts should be = weights(): " + str(coreset.wts) + " " + str(coreset.weights())


def test_build():
  for N in [0, 1, 10]:
    coreset = FullDataCoreset(N)
    for m in [1, 10, 100]:
      M = coreset.build(m)
      assert coreset.error() < tol, "FullDataCoreset failed: error not 0"
      assert np.all(coreset.weights() == np.ones(N)), "FullDataCoreset failed: weights not ones"
      assert M == N and coreset.M == N, "FullDataCoreset failed: M should = N after build. M = " + str(M) + " N = " + str(N) + " coreset.M = " +str(coreset.M)


