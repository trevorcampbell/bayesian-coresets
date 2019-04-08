from bayesiancoresets.base import IterativeCoreset
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(100)
tol = 1e-9

class DummyIterativeCoreset(IterativeCoreset):

  def _step(self):
    return True


def test_initialization():
  for N in [0, 1, 10]:
    coreset = DummyIterativeCoreset(N)
    assert coreset.N == N, "IterativeCoreset failed: N was not set properly"
    assert coreset.initialized, "IterativeCoreset failed: did not initialize"
    assert coreset.wts.shape[0] == coreset.N, "IterativeCoreset failed: probabilities do not sum to 1: sum = " + str(coreset.ps.sum())


def test_reset():
  for N in [0, 1, 10]:
    coreset = DummyIterativeCoreset(N)
    for m in [1, 10, 100]:
      coreset.build(m)
      #check reset
      coreset.reset()
      assert coreset.M == 0 and np.all(np.fabs(coreset.wts) == 0.) and not coreset.reached_numeric_limit, "IterativeCoreset failed: reset() did not properly reset"

def test_implementations():      
  for N in [0, 1, 10]:
    coreset = DummyIterativeCoreset(N)
    try:
      a = coreset.error()
      a = coreset.weights()
      a = coreset._step()
    except NotImplementedError as e:
      pass
    except:
      assert False, "IterativeCoreset shouldn't implement error, weights, _step"

def test_build():
  for N in [0, 1, 10]:
    coreset = DummyIterativeCoreset(N)
    for m in [1, 10, 100]:
      coreset.build(m)
      assert coreset.M == m or (coreset.M == 0 and coreset.N == 0), "IterativeCoreset failed: M should always be number of steps taken"
      assert np.all(coreset.wts >= 0), "IterativeCoreset failed: weights must be nonnegative"
      assert (coreset.wts > 0).sum() <= m, "IterativeCoreset failed: number of nonzero weights must be <= M: number = " + str((coreset.wts > 0).sum()) + " M = " + str(coreset.M)


