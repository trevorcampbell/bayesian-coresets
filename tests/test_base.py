import bayesiancoresets as bc
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(100)
tol = 1e-9

n_trials = 10
anms = ['Full', 'IS', 'Uniform']
algs = [bc.FullDataCoreset, bc.ImportanceSamplingCoreset, bc.UniformSamplingCoreset]
algs_nms = zip(anms, algs)
tests = [(N, algn) for N in [0, 1, 10] for algn in algs_nms]


def coreset_single(N, algn):
  anm, alg = algn
  coreset = alg(N)
  for m in [1, 10, 100]:
    fd.build(m)
    assert fd.error() < tol, "full wts failed: error not 0"
    assert np.all(fd.weights() == np.ones(x.shape[0])), "full wts failed: weights not ones"
  #check reset
  fd.reset()
  assert fd.M == 0 and np.all(np.fabs(fd.weights()) == 0.) and np.fabs(fd.error()) < tol and not fd.reached_numeric_limit, "FullDataset failed: reset() did not properly reset"

def test_coreset():
  for N, alg in tests:
    for n in range(n_trials):
      yield coreset_single, N, alg


