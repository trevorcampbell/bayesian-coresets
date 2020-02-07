import numpy as np
from ..util.errors import NumericalPrecisionError

class Projector(object):
    def project(self, pts, grad=False):
        raise NotImplementedError

    def update(self, wts, pts):
        raise NotImplementedError


class BlackBoxProjector(Projector):
    def __init__(self, sampler, projection_dimension, loglikelihood, grad_loglikelihood = None):
        self.sampler = sampler
        self.loglikelihood = loglikelihood
        self.grad_loglikelihood = grad_loglikelihood

    def project(self, pts, grad=False):
        lls = self.loglikelihood(pts, self.samples)
        lls -= lls.mean(axis=1)
        if grad:
            if self.grad_loglikelihood is None:
                raise ValueError('grad_loglikelihood was requested but not initialized in BlackBoxProjector.project')
            glls = self.gradloglikelihood(pts, self.samples)
            glls -= glls.mean(axis=2)[:, :, np.newaxis]
            return lls, glls
        else:
            return lls

    def update(self, wts, pts):
        self.samples = self.sampler(projection_dimension, wts, pts)
