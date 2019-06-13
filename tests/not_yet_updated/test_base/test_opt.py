from bayesiancoresets.base import OptimizationCoreset
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(100)
tol = 1e-9

class DummyOptimizationCoreset(OptimizationCoreset):

  def __init__(self, N):
    super().__init__(N=N)

  def _optimize(self):
    return np.zeros(self.N)

  def _max_reg_coeff(self):
    return 1.

def test_api():      
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

