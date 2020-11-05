import numpy as np
from ..util.errors import NumericalPrecisionError
from .snnls import SparseNNLS


class ImportanceSampling(SparseNNLS):

  def __init__(self, A, b):
      """
      Initialize the b

      Args:
          self: (todo): write your description
          A: (int): write your description
          b: (int): write your description
      """
    super().__init__(A, b)
    self.cts = np.zeros(self.w.shape[0]) 
    self.ps = np.sqrt((self.A**2).sum(axis=0))
    if np.any(self.ps > 0):
      self.ps /= self.ps.sum()
    else:
      self.ps = np.ones(self.w.shape[0]) / float(self.w.shape[0])
    self.check_error_monotone = False

  def reset(self):
      """
      Reset the progress bars.

      Args:
          self: (todo): write your description
      """
    super().reset()
    self.cts = np.zeros(self.w.shape[0]) 

  def _compute_sampling_probabilities(self):
      """
      Compute the probability probabilities.

      Args:
          self: (todo): write your description
      """
    self.ps = np.sqrt((self.A**2).sum(axis=0))
    if np.any(self.ps > 0):
      self.ps / self.ps.sum()

  def _select(self):
      """
      Return a copy of the selection.

      Args:
          self: (todo): write your description
      """
    return np.random.choice(self.ps.shape[0], p = self.ps)

  def _reweight(self, f):
      """
      R compute the weight function.

      Args:
          self: (todo): write your description
          f: (array): write your description
      """
    self.cts[f] += 1
    self.w = (self.cts / self.cts.sum()) / self.ps 

class UniformSampling(ImportanceSampling):
  def __init__(self, A, b):
      """
      Initialize b

      Args:
          self: (todo): write your description
          A: (int): write your description
          b: (int): write your description
      """
    super().__init__(A, b)
    self.ps = np.ones(self.w.shape[0]) / float(self.w.shape[0])
