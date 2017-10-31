import hilbertcoresets as hc
import numpy as np

def test_empty():
  x = np.zeros((0, 0))
  fd = hc.FullDataset(x)
  assert fd.error() == 0
  assert fd.weights() == np.ones(x.shape[0])
  for m in [0, 1, 10]:
    fd.run(m)
    assert fd.error() == 0
    assert fd.weights() == np.ones(x.shape[0])


def test_one():
  x = np.zeros((1, 3))
  fd = hc.FullDataset(x)
  assert fd.error() == 0
  assert fd.weights() == np.ones(x.shape[0])
  for m in [0, 1, 10]:
    fd.run(m)
    assert fd.error() == 0
    assert fd.weights() == np.ones(x.shape[0])

def test_many():
  x = np.zeros((10, 3))
  fd = hc.FullDataset(x)
  assert fd.error() == 0
  assert fd.weights() == np.ones(x.shape[0])
  for m in [0, 1, 10]:
    fd.run(m)
    assert fd.error() == 0
    assert fd.weights() == np.ones(x.shape[0])

