import numpy as np
from .util.errors import NumericalPrecisionError

class Projector(object):
    def project(self, pts, grad=False):
        """
        Projects the given points.

        Args:
            self: (todo): write your description
            pts: (array): write your description
            grad: (array): write your description
        """
        raise NotImplementedError

    def update(self, wts, pts):
        """
        Update a list of the given points.

        Args:
            self: (todo): write your description
            wts: (array): write your description
            pts: (array): write your description
        """
        raise NotImplementedError

class BlackBoxProjector(Projector):
    def __init__(self, sampler, projection_dimension, loglikelihood, grad_loglikelihood = None):
        """
        Initialize the gradient.

        Args:
            self: (todo): write your description
            sampler: (todo): write your description
            projection_dimension: (str): write your description
            loglikelihood: (todo): write your description
            grad_loglikelihood: (todo): write your description
        """
        self.projection_dimension = projection_dimension
        self.sampler = sampler
        self.loglikelihood = loglikelihood
        self.grad_loglikelihood = grad_loglikelihood
        self.update(np.array([]), np.array([]))

    def project(self, pts, grad=False):
        """
        Compute the log - likelihood.

        Args:
            self: (todo): write your description
            pts: (array): write your description
            grad: (array): write your description
        """
        lls = self.loglikelihood(pts, self.samples)
        lls -= lls.mean(axis=1)[:,np.newaxis]
        if grad:
            if self.grad_loglikelihood is None:
                raise ValueError('grad_loglikelihood was requested but not initialized in BlackBoxProjector.project')
            glls = self.grad_loglikelihood(pts, self.samples)
            glls -= glls.mean(axis=2)[:, :, np.newaxis]
            return lls, glls
        else:
            return lls

    def update(self, wts, pts):
        """
        Update the sampler.

        Args:
            self: (todo): write your description
            wts: (array): write your description
            pts: (array): write your description
        """
        self.samples = self.sampler(self.projection_dimension, wts, pts)
