from bayesiancoresets.base import SingleGreedyCoreset
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(100)
tol = 1e-9

class DummySingleGreedyCoreset(SingleGreedyCoreset):

  def __init__(self, N):
    super().__init__(N=N)

  def _search(self):
    return np.random.randint(self.N)

  def _step_coeffs(self, f):
    return 1., 1.

def test_initialization():
  for N in [0, 1, 10]:
    coreset = DummySingleGreedyCoreset(N)
    assert coreset.N == N, "SingleGreedyCoreset failed: N was not set properly"
    assert coreset.wts.shape[0] == coreset.N, "SingleGreedyCoreset failed: weights not initialized to size N"


def test_reset():
  for N in [0, 1, 10]:
    coreset = DummySingleGreedyCoreset(N)
    for m in [1, 10, 100]:
      coreset.build(m)
      #check reset
      coreset.reset()
      assert coreset.M == 0 and np.all(np.fabs(coreset.wts) == 0.) and not coreset.reached_numeric_limit, "SingleGreedyCoreset failed: reset() did not properly reset"

def test_implementations():      
  for N in [0, 1, 10]:
    coreset = DummySingleGreedyCoreset(N)
    try:
      a = coreset.error()
      a = coreset.weights()
      a = coreset._search()
      a = coreset._step_coeffs()
    except NotImplementedError as e:
      pass
    except:
      assert False, "SingleGreedyCoreset shouldn't implement error, weights, _search, _step_coeffs"

def test_build():
  for N in [0, 1, 10]:
    coreset = DummySingleGreedyCoreset(N)
    for m in [1, 10, 100]:
      coreset.build(m)
      assert coreset.M == m or (coreset.M == 0 and coreset.N == 0), "SingleGreedyCoreset failed: M should always be number of steps taken"
      assert np.all(coreset.wts >= 0), "SingleGreedyCoreset failed: weights must be nonnegative"
      assert (coreset.wts > 0).sum() <= m, "SingleGreedyCoreset failed: number of nonzero weights must be <= M: number = " + str((coreset.wts > 0).sum()) + " M = " + str(coreset.M)

