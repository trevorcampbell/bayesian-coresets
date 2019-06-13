from bayesiancoresets.base import IterativeCoreset
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(100)
tol = 1e-9

class DummyIterativeCoreset(IterativeCoreset):

  def __init__(self, N):
    super().__init__(N=N)

  def _step(self):
    return True

def test_api():      
  for N in [0, 1, 10]:
    coreset = DummyIterativeCoreset(N)
    try:
      a = coreset.error()
      a = coreset.weights()
      a = coreset._step()
    except NotImplementedError as e:
      pass
    except:
      assert False, coreset.alg_name+ " shouldn't implement error, weights, _step"

