import numpy as np


class FrankWolfe(object):
  def __init__(self):
    self.reset()

  def run(self, M):
    if M == 0:
      return

    if self.M == 0:
      self.wts = np.zeros(self.N)
      xs = self.x.sum(axis=0)
      scores = (xs*self.x).sum(axis=1)/self.norms
      f = scores.argmax()
      wts[f] = self.sig/self.norms[f]
      self.M = 1

    for m in range(self.M, M):
      xs = ((1. - self.wts)[:, np.newaxis]*self.x).sum(axis=0)
      scores = (xs*self.x).sum(axis=1)/self.norms
      f = scores.argmax()
      
      xw = (self.wts[:,np.newaxis]*self.x).sum(axis=0)
      gammanum = self.sig*scores[f]
      gammadenom = self.sig**2
      for nz in self.wts.nonzero()[0]:
        gammanum -= self.wts[nz]*(self.x[nz, :].dot(xs)) 
        gammadenom -= 2.*self.wts[nz]*self.sig/self.norms[f]*(self.x[f, :].dot(self.x[nz, :]))
        gammadenom += self.wts[nz]*(self.x[nz,:].dot(xw))
      gamma = gammanum/gammadenom
      if gamma < 0 or gamma > 1:
        print 'Warning: gamma not in [0, 1]: ' + str(gamma)
        gamma = (0. if gamma < 0 else gamma)
        gamma = (1. if gamma > 1 else gamma)
      self.wts *= (1.-gamma)
      self.wts[f] += gamma*self.sig/self.norms[f] 

    self.M = M
    return

  def reset(self):
    self.M = 0
    self.wts = None

class SketchedFrankWolfe(FrankWolfe):
  
  def __init__(self, data, gradlogp, dim, sketch_dim, weighting_dist):
    self.norms = np.sqrt((v**2).sum(axis=1))
    self.sig = norms.sum()
    self.N = self.x.shape[0]
    self.dim = 0

  def get_weighting_dist(self):
    pass

  def update_sketch_dimension(self, sketch_dim):
    if sketch_dim < self.x.shape[1]:
      self.x = self.x[:, :sketch_dim]

    if sketch_dim > self.x.shape[1]:
       old_dim = self.x.shape[1]
       w = np.zeros((self.N, sketch_dim))
       w[:, :old_dim] = self.x
       w *= np.sqrt(old_dim)
       for j in range(sketch_dim-old_dim):
         w[:, j+old_dim] = np.sqrt(self.dim)*self.sample_sketch_component(XXX)
       w /= np.sqrt(sketch_dim)
       self.x = w.copy()

    return

  def sample_sketch_component(self):
    pass
 

class ImportanceSampling(object):

  def __init__(self):
    self.reset()

  def run(self, M):
    if not self.ps:
      self.ps = self.norms/self.norms.sum()
      self.cts = np.zeros(self.N)
      
    self.cts += np.random.multinomial(M - self.M, ps)
    self.wts = self.cts/self.ps/M
    
    self.M = M
    return

  def reset(self):
    self.M = 0
    self.wts = None
    self.cts = None
    self.ps = None


class SketchedImportanceSampling(ImportanceSampling):

  def __init__(self, gradlogp, weighting_dist):
    pass

