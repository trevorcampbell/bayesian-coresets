from bayesiancoresets.base import OptimizationCoreset
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(100)
tol = 1e-9

class DummyOptimizationCoreset(OptimizationCoreset):

  def _optimize(self):
    return np.zeros(self.N)

  def _max_reg_coeff(self):
    return 1.


def test_initialization():
  for N in [0, 1, 10]:
    coreset = DummyOptimizationCoreset(N)
    assert coreset.N == N, "OptimizationCoreset failed: N was not set properly"
    assert coreset.initialized, "OptimizationCoreset failed: did not initialize"
    assert coreset.wts.shape[0] == coreset.N, "OptimizationCoreset failed: probabilities do not sum to 1: sum = " + str(coreset.ps.sum())


def test_reset():
  for N in [0, 1, 10]:
    coreset = DummyOptimizationCoreset(N)
    for m in [1, 10, 100]:
      coreset.build(m)
      #check reset
      coreset.reset()
      assert coreset.M == 0 and np.all(np.fabs(coreset.wts) == 0.) and not coreset.reached_numeric_limit, "OptimizationCoreset failed: reset() did not properly reset"

def test_implementations():      
  for N in [0, 1, 10]:
    coreset = DummyOptimizationCoreset(N)
    try:
      a = coreset.error()
      a = coreset.weights()
      a = coreset._optimize()
    except NotImplementedError as e:
      pass
    except:
      assert False, "OptimizationCoreset shouldn't implement error, weights, _step"

def test_build():
  for N in [0, 1, 10]:
    coreset = DummyOptimizationCoreset(N)
    for m in [1, 10, 100]:
      coreset.build(m)
      assert coreset.M == m or (coreset.M == 0 and coreset.N == 0), "OptimizationCoreset failed: M should always be number of steps taken"
      assert np.all(coreset.wts >= 0), "OptimizationCoreset failed: weights must be nonnegative"
      assert (coreset.wts > 0).sum() <= m, "OptimizationCoreset failed: number of nonzero weights must be <= M: number = " + str((coreset.wts > 0).sum()) + " M = " + str(coreset.M)


