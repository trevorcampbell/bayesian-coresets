import numpy as np

def gaussian_potentials(Siginv, xSiginvx, xSiginv, logdetSig, x, samples):
  return -x.shape[1]/2*np.log(2*np.pi) - 1./2.*logdetSig - 1./2.*(xSiginvx[:, np.newaxis] - 2.*np.dot(xSiginv, samples.T) + (np.dot(samples, Siginv)*samples).sum(axis=1))
 
np.set_printoptions(linewidth=10000000)

muw = np.random.randn(5)
Sigw = np.random.randn(5, 5)
Sigw = Sigw.dot(Sigw.T)

Sig = np.random.randn(5,5)
Sig = Sig.dot(Sig.T)
SigInv = np.linalg.inv(Sig)

x = np.random.multivariate_normal(np.random.randn(5)+np.ones(5), Sig, 2)

xSigInv = np.dot(x, SigInv)
xSigInvx = (x*xSigInv).sum(axis=1)
logdetSig = np.linalg.slogdet(Sig)[1]

SigL = np.linalg.cholesky(Sig)
SigLInv = np.linalg.inv(SigL)

nu = (x - muw).dot(SigLInv.T)
Psi = np.dot(SigLInv, np.dot(Sigw, SigLInv.T))

exact_cov = np.dot(nu, np.dot(Psi, nu.T)) + 0.5*np.trace(np.dot(Psi.T, Psi))



samps = np.random.multivariate_normal(muw, Sigw, 100000000)
empirical_cov = np.cov(gaussian_potentials(SigInv, xSigInvx, xSigInv, logdetSig, x, samps))


print(exact_cov)
print(empirical_cov)
