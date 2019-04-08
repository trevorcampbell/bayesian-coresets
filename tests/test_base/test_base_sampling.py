from bayesiancoresets.base import SamplingCoreset
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(100)
tol = 1e-9

class DummySamplingCoreset(SamplingCoreset):

  def _compute_sampling_probabilities(self):
    return np.ones(self.N)/self.N
  
  def _update_cache(self):
    pass

def test_initialization():
  for N in [0, 1, 10]:
    coreset = DummySamplingCoreset(N)
    assert coreset.N == N, "SamplingCoreset failed: N was not set properly"
    assert coreset.initialized, "SamplingCoreset failed: did not initialize"
    assert N == 0 or coreset.ps.sum() == 1., "SamplingCoreset failed: probabilities do not sum to 1: sum = " + str(coreset.ps.sum())
    assert coreset.ps.shape[0] == coreset.N and coreset.cts.shape[0] == coreset.N and coreset.wts.shape[0] == coreset.N, "SamplingCoreset failed: probabilities do not sum to 1: sum = " + str(coreset.ps.sum())

def test_reset():
  for N in [0, 1, 10]:
    coreset = DummySamplingCoreset(N)
    for m in [1, 10, 100]:
      coreset.build(m)
      #check reset
      coreset.reset()
      assert coreset.M == 0 and np.all(np.fabs(coreset.wts) == 0.) and not coreset.reached_numeric_limit, "SamplingCoreset failed: reset() did not properly reset"

def test_implementations():      
  for N in [0, 1, 10]:
    coreset = DummySamplingCoreset(N)
    try:
      a = coreset.error()
      a = coreset.weights()
    except NotImplementedError as e:
      pass
    except:
      assert False, "SamplingCoreset shouldn't implement error or weights"

def test_build():
  for N in [0, 1, 10]:
    coreset = DummySamplingCoreset(N)
    for m in [1, 10, 100]:
      coreset.build(m)
      assert (coreset.M == m and coreset.M == coreset.cts.sum()) or N == 0, "SamplingCoreset failed: M should always be number of samples taken: cts = " + str(coreset.cts.sum()) + " M = " + str(coreset.M)
      assert np.all(coreset.wts >= 0), "SamplingCoreset failed: weights must be nonnegative"
      assert (coreset.wts > 0).sum() <= m, "SamplingCoreset failed: number of nonzero weights must be <= M: number = " + str((coreset.wts > 0).sum()) + " M = " + str(coreset.M)


