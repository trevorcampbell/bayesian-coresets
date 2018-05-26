import bayesiancoresets as bc
import numpy as np
import warnings


warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.random.seed(100)
tol = 1e-12

def test_empty():
  x = np.random.randn(0, 0)
  fd = bc.FullDataset(x)
  for m in [1, 10, 100]:
    fd.run(m)
    assert fd.error() < tol, "full wts failed: error not 0"
    assert np.all(fd.weights() == np.ones(x.shape[0])), "full wts failed: weights not ones"
  #check reset
  fd.reset()
  assert fd.M == 0 and np.all(np.fabs(fd.weights()) == 0.) and np.fabs(fd.error() - np.sqrt((fd.snorm**2).sum())) < tol and not fd.reached_numeric_limit, "FullDataset failed: reset() did not properly reset"



def test_one():
  x = np.random.randn(1, 3)
  fd = bc.FullDataset(x)
  for m in [1, 10, 100]:
    fd.run(m)
    assert fd.error() < tol, "full wts failed: error not 0"
    assert np.all(fd.weights() == np.ones(x.shape[0])), "full wts failed: weights not ones: "+str(fd.weights())
  #check reset
  fd.reset()
  assert fd.M == 0 and np.all(np.fabs(fd.weights()) == 0.) and np.fabs(fd.error() - np.sqrt((fd.snorm**2).sum())) < tol and not fd.reached_numeric_limit, "FullDataset failed: reset() did not properly reset"


def test_many():
  x = np.random.randn(10, 3)
  fd = bc.FullDataset(x)
  for m in [1, 10, 100]:
    fd.run(m)
    assert fd.error() < tol, "full wts failed: error not 0"
    assert np.all(fd.weights() == np.ones(x.shape[0])), "full wts failed: weights not ones "+str(fd.weights())
  #check reset
  fd.reset()
  assert fd.M == 0 and np.all(np.fabs(fd.weights()) == 0.) and np.fabs(fd.error() - np.sqrt((fd.snorm**2).sum())) < tol and not fd.reached_numeric_limit, "FullDataset failed: reset() did not properly reset"


