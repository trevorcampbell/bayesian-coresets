import bayesiancoresets as bc
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(100)
tol = 1e-9

n_trials = 10
anms = ['Full']
algs = [bc.FullDataCoreset]
algs_nms = zip(anms, algs)
tests = [(N, algn) for N in [0, 1, 10] for algn in algs_nms]


def test_full_data(N, algn):
  for N in [0, 1, 10]:
    coreset = bc.FullDataCoreset(N)
    for m in [1, 10, 100]:
      fd.build(m)
      assert coreset.error() < tol, "full wts failed: error not 0"
      assert np.all(coreset.weights() == np.ones(x.shape[0])), "full wts failed: weights not ones"
    #check reset
    coreset.reset()
    assert coreset.M == N and np.all(np.fabs(fd.weights() - 1.) == 0.) and np.fabs(fd.error()) < tol and not fd.reached_numeric_limit, "FullDataset failed: reset() did not properly reset"
