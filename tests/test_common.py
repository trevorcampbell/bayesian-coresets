import numpy as np
import warnings
from abstract_dummies import *
from bayesiancoresets import *

warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(100)
tol = 1e-9

n_trials = 5
algs = [FullDataCoreset]
types = ['dummy']
algs.extend([DummyOptimizationCoreset, DummyIterativeCoreset, DummySingleGreedyCoreset, DummySamplingCoreset])
types.extend(['dummy']*4)
algs.extend([L1KLCoreset, GreedyKLCoreset, SamplingKLCoreset, UniformSamplingKLCoreset])
types.extend(['kl']*4)
algs.extend([LassoCoreset, FrankWolfeCoreset, GIGACoreset, ForwardStagewiseCoreset, MatchingPursuitCoreset, LARCoreset, VectorSamplingCoreset, VectorUniformSamplingCoreset, OrthoPursuitCoreset])
types.extend(['vector']*9)
tests = [(N, D, at) for N in [0, 1, 10] for D in [1, 3, 10] for at in zip(algs, types)]



def test_coreset():
  for N, D, at in tests:
    alg, typ = at
    if typ == 'dummy':
      coreset = alg(N)
    elif typ == 'kl':
      coreset = alg(potentials=lambda x : dummy_potentials(x, N) , sampler=lambda w, n : dummy_sampler(w, n, D), n_samples=20, N=N)
    elif typ == 'vector':
      x = np.random.randn(N, D)
      coreset = alg(x)
    else:
      raise ValueError('Unexpected coreset alg type')

    for n in range(n_trials):
      yield initialization, N, D, coreset
      yield reset, N, D, coreset
      yield build, N, D, coreset
      yield input_validation, N, D, coreset

def input_validation(N, D, coreset):
  pass

def initialization(N, D, coreset):
  assert coreset.N == N, coreset.alg_name + " failed: N was not set properly"
  assert coreset.wts.shape[0] == coreset.N, coreset.alg_name + " failed: wts not initialized to size N"


def reset(N, D, coreset):
  for m in [1, 10]:
    coreset.build(m)
    #check reset
    coreset.reset()
    assert coreset.M == 0 and np.all(np.fabs(coreset.wts) == 0.) and not coreset.reached_numeric_limit, coreset.alg_name + " failed: reset() did not properly reset"

def build(N, D, coreset):
  for m in [1, 10]:
    coreset.build(m)
    #assert coreset.M <= m or (coreset.M == 0 and coreset.N == 0), coreset.alg_name + " failed: M should always be number of steps taken"
    assert np.all(coreset.wts >= 0), coreset.alg_name + " failed: weights must be nonnegative"
    assert (coreset.wts > 0).sum() <= coreset.M, coreset.alg_name +  " failed: number of nonzero weights must be <= M: number = " + str((coreset.wts > 0).sum()) + " M = " + str(coreset.M)
    assert (coreset.wts > 0).sum() == coreset.size(), coreset.alg_name +  " failed: number of nonzero weights != size()"
