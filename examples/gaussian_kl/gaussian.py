import autograd.numpy as np
from autograd import grad
import warnings
from scipy.stats import multivariate_normal as mvn


def gaussian_sampler(th0, Sig0inv, Siginv, x, w, n_samples):
  muw, Sigw = weighted_post(th0, Sig0inv, Siginv, x, w)
  return np.random.multivariate_normal(muw, Sigw, n_samples)

def gaussian_potentials(Siginv, xSiginvx, xSiginv, logdetSig, x, samples):
  return -x.shape[1]/2*np.log(2*np.pi) - 1./2.*logdetSig - 1./2.*(xSiginvx[:, np.newaxis] - 2.*np.dot(xSiginv, samples.T) + (np.dot(samples, Siginv)*samples).sum(axis=1))
  

#NB: without constant terms
#E[Log N(x; mu, Sig)] under mu ~ N(muw, Sigw)
def ll_m1_exact(muw, Sigw, Siginv, x, constant=True):
  return (-0.5*x.shape[1]*np.log(2*np.pi) + 0.5*np.linalg.slogdet(Siginv)[1] if constant else 0.) -0.5*np.trace(np.dot(Siginv, Sigw)) -0.5*(np.dot((x - muw), Siginv)*(x-muw)).sum(axis=1)

#Covar[Log N(x; mu, Sig)] under mu ~ N(muw, Sigw)
def ll_m2_exact(muw, Sigw, Siginv, x):
  L = np.linalg.cholesky(Siginv)
  Rho = np.dot(np.dot(L.T, Sigw), L)

  crho = 2*(Rho**2).sum() + (np.diag(Rho)*np.diag(Rho)[:, np.newaxis]).sum()

  mu = np.dot(L.T, (x - muw).T).T
  musq = (mu**2).sum(axis=1)

  return 0.25*(crho + musq*musq[:, np.newaxis] + np.diag(Rho).sum()*(musq + musq[:,np.newaxis]) + 4*np.dot(np.dot(mu, Rho), mu.T))

#Var[Log N(x;, mu, Sig)] under mu ~ N(muw, Sigw)
def ll_m2_exact_diag(muw, Sigw, Siginv, x):
  L = np.linalg.cholesky(Siginv)
  Rho = np.dot(np.dot(L.T, Sigw), L)

  crho = 2*(Rho**2).sum() + (np.diag(Rho)*np.diag(Rho)[:, np.newaxis]).sum()

  mu = np.dot(L.T, (x - muw).T).T
  musq = (mu**2).sum(axis=1)

  return 0.25*(crho + musq**2 + 2*np.diag(Rho).sum()*musq + 4*(np.dot(mu, Rho)*mu).sum(axis=1))
 
if __name__ == '__main__':
  for t in range(100):
    N_samps = 10000000
    D = 5
    N_data = 3
    x = np.random.randn(N_data, D)
    mu = np.random.randn(D)
    Sig = np.diag(np.random.randn(D)**2)
    SigL = np.eye(D)
    samples = np.random.multivariate_normal(mu, Sig, N_samps)
  
    
    lls = np.zeros((N_data, N_samps))
    for i in range(N_data):
      lls[i, :] = gaussian_logpdf(x[i, :], samples, SigL)
    #print('Sig vs SigL err : ' + str(   np.sqrt( (((np.linalg.inv(SigL).dot(SigL)) - np.eye(D))**2).sum() )))
    #print('emp: ' + str(lls.mean(axis=1)))
    #print('ex: ' + str(exact_mean))
    #print('mean err: ' + str(np.sqrt( ((lls.mean(axis=1) - exact_mean)**2).sum() ) / np.sqrt( ((exact_mean)**2).sum()) ))
      
    #print('ll err: ' + str( np.sqrt(((lls-lls2)**2).sum()) ))
    exact_mean = ll_m1_exact(mu, Sig, np.linalg.inv(SigL), x)
    mean_formula = -x.shape[1]*0.5*np.log(2*np.pi) - 0.5*( (x - mu)**2).sum(axis=1) - 0.5*Sig.sum()
    empirical_mean = np.mean(lls, axis=1)
    print('exact mean: ' + str(exact_mean))
    print('mean formula: ' + str(mean_formula))
    print('empirical mean: ' + str(empirical_mean))

    empirical_var = np.diagonal(np.cov(lls))
    var_formula = 0.5*(Sig**2).sum() + (((x-mu)**2)*np.diagonal(Sig)).sum(axis=1)
    exact_diag = ll_m2_exact_diag(mu, Sig, np.linalg.inv(SigL), x) 
    exact_diag_cov = np.diagonal(ll_m2_exact(mu, Sig, np.linalg.inv(SigL), x))
    exact_diag_minus = ll_m2_exact_diag(mu, Sig, np.linalg.inv(SigL), x) - ll_m1_exact(mu, Sig, np.linalg.inv(SigL), x, constant=False)**2
    print('m2 exact diag: ' + str(exact_diag))
    print('m2 diag of exact cov: ' + str(exact_diag_cov))
    print('diag minus m1**2 : ' + str(exact_diag_minus))
    print('var formula: ' + str(var_formula))
    print('empirical var: ' + str(empirical_var))


