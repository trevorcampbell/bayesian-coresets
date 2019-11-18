import numpy as np
from scipy.optimize import nnls
from ..util.errors import NumericalPrecisionError
from .snnls import SparseNNLS

class LAR(SparseNNLS):

  def __init__(self, A, b):
    raise NotImplementedError
    super().__init__(A, b)
    self.active_idcs = np.zeros(self.w.shape[0], dtype=np.bool)
    residual = self.b - self.A.dot(self.w)
    self.active_idcs[(self.An.T.dot(residual)).argmax()] = True

    Anorms = np.sqrt((self.A**2).sum(axis=0))
    if np.any( Anorms == 0):
      raise ValueError(self.alg_name+'.__init__(): A must not have any 0 columns')
    self.An = self.A / Anorms

    self.bnorm = np.sqrt(((self.b)**2).sum())
    if self.bnorm == 0.:
      raise NumericalPrecisionError('norm of b must be > 0')
    self.bn = self.b / self.bnorm

  def reset(self):
    super().reset()
    self.active_idcs = np.zeros(self.w.shape[0], dtype=np.bool)
    residual = self.b - self.A.dot(self.w)
    self.active_idcs[(self.An.T.dot(residual)).argmax()] = True

  def select(self):
    #do least squares on active set
    res = nnls(self.A[:, self.active_idcs].T, self.b, maxiter=100*self.A.shape[1])

    x_opt = self.A.dot(res[0])
    w_opt = np.zeros(self.w.shape[0])
    w_opt[self.active_idcs] = res[0]
    xw = self.A.dot(self.w)
    sdir = x_opt - xw
    sdir /= np.sqrt((sdir**2).sum())

    #do line search towards x_opt

    #find earliest gamma for which a variable joins the active set
    #anywhere gamma_denom = 0 or gamma < 0, the variable never enters the active set 
    gamma_nums = (sdir - self.x).dot(self.b - xw)
    gamma_denoms = (sdir - self.x).dot(x_opt - xw)
    good_idcs = np.logical_not(np.logical_or(gamma_denoms == 0, gamma_nums*gamma_denoms < 0))
    gammas = np.inf*np.ones(gamma_nums.shape[0])
    gammas[good_idcs] = gamma_nums[good_idcs]/gamma_denoms[good_idcs]
    f_least_angle = gammas.argmin()
    gamma_least_angle = gammas[f_least_angle]

    #find earliest gamma for which a variable leaves the active set
    f_leave_active = -1
    gamma_leave_active = np.inf
    gammas[:] = np.inf
    leave_idcs = w_opt < 0
    gammas[leave_idcs] = self.wts[leave_idcs]/(self.wts[leave_idcs] - w_opt[leave_idcs])
    f_leave_active = gammas.argmin()
    gamma_leave_active = gammas[f_leave_active]


  def _reweight(self):
    if gamma_leave_active >= 1. and gamma_least_angle >= 1.:
      #no variable leaves active set, and no variable becomes more aligned; we are done
      self.xw = x_opt
      self.wts = w_opt
      self.reached_numeric_limit = True
    elif gamma_leave_active < gamma_least_angle:
      #a variable leaves the active set first
      self.wts = (1. - gamma_leave_active)*self.wts + gamma_leave_active*w_opt
      self.wts[f_leave_active] = 0
      self.active_idcs[f_leave_active] = False
      if self.use_cached_xw:
        self.xw = (1. - gamma_leave_active)*self.xw + gamma_leave_active*x_opt
      else:
        self.xw = self.wts.dot(self.x)
    else: 
      #a variable becomes aligned first, joins active set
      self.wts = (1. - gamma_least_angle)*self.wts + gamma_least_angle*w_opt
      self.active_idcs[f_least_angle] = True
      if self.use_cached_xw:
        self.xw = (1. - gamma_least_angle)*self.xw + gamma_least_angle*x_opt
      else:
        self.xw = self.wts.dot(self.x)

    return True


