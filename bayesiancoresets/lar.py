import numpy as np
from .coreset import CoresetConstruction

class LAR(CoresetConstruction):

  def _xw_unscaled(self):
    return False
  
  def _initialize(self):
    f = self._search()
    self.search_x = self.x[f, :]
    self.search_w = np.zeros(self.wts.shape[0])
    self.search_w[f] = 1.

  def _step(self, use_cached_xw):
    #move along the search direction until another variable is as aligned *or* a current variable leaves the active set
    #TODO

    #update the weights
    #TODO

    #use LBFGS to compute next search direction
    X = self.x[self.wts > 0, :]
    w0 = self.wts[self.wts > 0]
    res = minimize(fun = lambda w : ((self.snorm*self.xs - w.dot(X))**2).sum(), 
             x0 = w0, method='L-BFGS-B', 
             jac = lambda w : (w.dot(X)).dot(X.T) - 2*self.snorm*self.xs.dot(X.T), 
             options ={'ftol': 1e-12, 'gtol': 1e-9})
 
    #if the optimizer failed or our cost increased, stop
    xopt = res.x.dot(X)
    if not res.success or np.sqrt(((self.snorm*self.xs - xopt)**2).sum()) >= self.error():
      self.reached_numeric_limit = True
      return

    wopt = np.zeros(self.wts.shape[0])
    wopt[self.wts > 0] = res.x

    #update search direction
    self.search_x = xopt - self.xw
    self.search_x /= np.sqrt((self.search_dir**2).sum())
    

    #update weights, xw, and prev_cost
    self.wts[self.wts > 0] = res.x
    self.xw = self.wts.dot(self.x)
    self.prev_cost = self.error()

  def _search(self):
    return (((self.snorm*self.xs - self.xw)*self.x).sum(axis=1)).argmax()


