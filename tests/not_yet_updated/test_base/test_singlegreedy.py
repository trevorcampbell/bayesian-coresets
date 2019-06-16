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

def test_api():      
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

