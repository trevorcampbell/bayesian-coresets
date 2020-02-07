import numpy as np
from ..util.errors import NumericalPrecisionError

class Projector(object):
    def project(self, pts, grad=False):
        raise NotImplementedError

    def update(self, wts, pts):
        raise NotImplementedError


class BlackBoxProjector(Projector):
    def __init__(self, sampler, loglikelihood, grad_loglikelihood):
        self.sampler = sampler
        self.loglikelihood = loglikelihood
        self.grad_loglikelihood = grad_loglikelihood

    def project(self, pts, grad=False):
        pass

    def update(self, wts, pts):
        pass
