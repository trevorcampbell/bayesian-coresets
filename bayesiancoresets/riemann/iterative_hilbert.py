import numpy as np
from ..base.iterative import IterativeCoreset
from ..riemann.kl import KLCoreset
from ..tangent import FixedFiniteTangentSpace
from ..hilbert import GIGACoreset

class IterativeHilbertCoreset(KLCoreset,IterativeCoreset):

    def __init__(self, N, tangent_space_factory, hilbert_coreset_class = GIGACoreset, step_sched = lambda i : 1./(1.+i), optimizing = True):
      super().__init__(N=N) 
      self.tsf = tangent_space_factory
      self.step_sched = step_sched
      self.hilbert_coreset_class = hilbert_coreset_class
      self.optimizing = optimizing

    def _terminate_on_size(self):
      return False

    def _step(self, sz, itr):
      if self.size() == 0:
        #on init, seed with a uniform subsample
        #TODO make this more memory efficient with sparse weights/idcs
        seeding_wts = (self.N/M) * np.random.multinomial(M, [1/self.N]*self.N)
        seeding_idcs = np.arange(self.N)[seeding_wts > 0]
        seeding_wts = seeding_wts[seeding_wts > 0]
        self._overwrite(seeding_idcs, seeding_wts)
    
      #build the tangent space
      tangent_space = self.tsf(self.wts, self.idcs)

      #scale the vectors to account for the upcoming stochastic update
      scaling_vector = self.step_sched(itr)*np.ones(tangent_space.num_vectors())
      scaling_vector[self.idcs] += ((1 - self.step_sched(itr)) * self.wts)

      #put the scaled vectors in their own new tangent space
      scaled_tangent_space= FixedFiniteTangentSpace(scaling_vector[:, np.newaxis]*tangent_space[:])

      #pass the scaled tangent space to hilbert coreset construction
      hilbert_coreset = self.hilbert_coreset_class(scaled_tangent_space)
      hilbert_coreset.build(sz, sz) #build a coreset of size sz

      #optimize if requested
      if self.optimizing:
          try:
              hilbert_coreset.optimize() 
          except RuntimeError:
              print("error optimizing!")#, ", matrix was ", hilbert_coreset.T[hilbert_coreset.idcs])
              #gained a small bit of accuracy just by terminating the algorithm here... since the case where nnls was terminating on iteration count was when our tangent space was basically zeroes. 
              #self.itrs = self.num_its
              #return
              #need to test eigenvalues here
              pass #need this line if I remove the print statement

      #update the weights 
      candidate_wts, candidate_idcs = hilbert_coreset.weights() 
      candidate_wts *= scaling_vector[candidate_idcs]
      self._overwrite(candidate_idcs, candidate_wts)

