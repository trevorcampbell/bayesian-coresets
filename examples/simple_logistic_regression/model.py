import numpy as np
#computes the logistic regression log-likelihood for data z and parameter th
#input: z = N x D numpy array, th = length D numpy array
#output: length N numpy array of log_likelihoods
def log_likelihood(z, th):
  if len(z.shape) == 1:
    m = -(th*z).sum()
    if m < 100:
      m = -np.log1p(np.exp(m))
    else:
      m = -m
    return m 
  else:
    m = -(th*z).sum(axis=1)
    idcs = m < 100
    m[idcs] = -np.log1p(np.exp(m[idcs]))
    m[np.logical_not(idcs)] = -m[np.logical_not(idcs)]
    return m

#computes the gradient of the logistic regression log-likelihood
# z is data, one row per vector; th is the parameter
#input: z = N x D numpy array, th = length D numpy array, idx = optional gradient component index
#output: (if idx = None): N x D array of gradients  (if idx = integer) N x 1 array of gradient components
def grad_log_likelihood(z, th, idx=None):
  if len(z.shape) == 1:
    m = -(th*z).sum()
    if m < 100:
      m = np.exp(m)/(1.+np.exp(m))
    else:
      m = 1.
    return m*z
  else:
    m = -(th*z).sum(axis=1)
    idcs = m < 100
    m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))
    m[np.logical_not(idcs)] = 1.
    if idx is None:
      return m[:, np.newaxis]*z
    return m*z[:, idx]

#computes the log prior for parameter th
#input: th = length D numpy array
#output: log prior density value, scalar
def log_prior(th):
  return -0.5*th.shape[0]*np.log(2.*np.pi) - 0.5*(th**2).sum()

#computes the log prior gradient for parameter th
#input: th = length D numpy array
#output: length D numpy gradient array
def grad_log_prior(th):
  return -th

#computes the log joint probability for data z and parameter th, where the data are weighted by wts
#input: Z = N x D numpy array, th = length D numpy array, wts = length N numpy array of nonnegative values
#output: weighted log joint, scalar
def log_joint(Z, th, wts):
  return (wts*log_likelihood(Z, th)).sum() + log_prior(th)

#same as above; outputs length D numpy array gradient
def grad_log_joint(z, th, wts):
  return grad_log_prior(th) + (wts[:, np.newaxis]*grad_log_likelihood(z, th)).sum(axis=0)

#same as above; outputs D x D numpy array Hessian
def hess_log_joint(z, th):
  es = np.exp(-(th*z).sum(axis=1))
  H_log_like = -(z.T).dot((es/(1.+es)**2)[:, np.newaxis]*z)
  H_log_prior = -np.eye(th.shape[0])
  return H_log_like + H_log_prior

def hess_log_joint_w(z, th, wts):
  #m = -(th*z).sum(axis=1)
  #idcs = m < 100
  #m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))
  #m[np.logical_not(idcs)] = 1.
  #H_log_like = -(z.T).dot((m**2)[:, np.newaxis]*z)
  es = np.exp(-(th*z).sum(axis=1))
  H_log_like = -(wts*z.T).dot((es/(1.+es)**2)[:, np.newaxis]*z)
  H_log_prior = -np.eye(th.shape[0])
  return H_log_like + H_log_prior


