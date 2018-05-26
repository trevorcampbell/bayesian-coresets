import numpy as np
from .coreset import IterativeCoresetConstruction
from scipy.optimize import lsq_linear

class LAR(IterativeCoresetConstruction):

  def _xw_unscaled(self):
    return False

  def _initialize(self):
    self.active_idcs = np.zeros(self.wts.shape[0], dtype=np.bool)
    self.active_idcs[self._search()] = True
  
  def _step(self, use_cached_xw):
    #do least squares on active set
    X = self.x[self.active_idcs, :]
    res = lsq_linear(X.T, self.snorm*self.xs, max_iter=max(1000, 10*self.xs.shape[0]))

    #if the optimizer failed or our cost increased, stop
    prev_cost = self.error()
    if not res.success or np.sqrt(2.*res.cost) >= prev_cost:
      self.reached_numeric_limit = True
      return False
  
    x_opt = res.x.dot(X)
    w_opt = np.zeros(self.wts.shape[0])
    w_opt[self.active_idcs] = res.x
    sdir = x_opt - self.xw
    sdir /= np.sqrt((sdir**2).sum())

    #do line search towards x_opt

    #find earliest gamma for which a variable joins the active set
    #anywhere gamma_denom = 0 or gamma < 0, the variable never enters the active set 
    gamma_nums = (sdir - self.x).dot(self.snorm*self.xs - self.xw)
    gamma_denoms = (sdir - self.x).dot(x_opt - self.xw)
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
      if use_cached_xw:
        self.xw = (1. - gamma_leave_active)*self.xw + gamma_leave_active*x_opt
      else:
        self.xw = self.wts.dot(self.x)
    else: 
      #a variable becomes aligned first, joins active set
      self.wts = (1. - gamma_least_angle)*self.wts + gamma_least_angle*w_opt
      self.active_idcs[f_least_angle] = True
      if use_cached_xw:
        self.xw = (1. - gamma_least_angle)*self.xw + gamma_least_angle*x_opt
      else:
        self.xw = self.wts.dot(self.x)

    return True

  def _search(self):
    return (((self.snorm*self.xs - self.xw)*self.x).sum(axis=1)).argmax()


