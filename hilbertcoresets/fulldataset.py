import numpy as np

class FullDataset(object):
  def __init__(self, x):
    self.wts = np.ones(x.shape[0])

  def run(self, M):
    return

  def reset(self):
    return

  def weights(self):
    return self.wts

  def error(self):
    return 0.


