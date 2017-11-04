import numpy as np
from hilbertcoresets.geometry import *

tol = 1e-9
n_trials = 50
Ds = range(1, 10)

def check_box_data(D):
  N = 2**D
  x = np.zeros((N, D))
  for i in range(N):
    for d in range(D):
      x[i, d] = (i >> d) % 2
  x[x == 0] = -1

  diam = compute_diam(x)
  assert np.fabs(diam -  2.) < tol, "diam = " + str(diam)

  normratio = compute_normratio(x)
  assert np.fabs(normratio - 1.) < tol, "nrmratio = " + str(normratio)

  nu, r = compute_nu(x, diam)
  assert np.fabs(r - N) < tol, "r = " + str(r) + "N = " + str(N)

def test_box():
  for D in Ds:
    yield check_box_data, D


def check_ball_data(N, D):
  x = np.random.normal(0., 1., (N/2+1, D))
  x /= np.sqrt((x**2).sum(axis=1))[:, np.newaxis]
  x = np.vstack((x, -x))
  N = x.shape[0]
  
  diam = compute_diam(x)
  assert np.fabs(diam -  2.) < tol, "diam = " + str(diam)

  normratio = compute_normratio(x)
  assert np.fabs(normratio - 1.) < tol, "nrmratio = " + str(normratio)

  nu, r = compute_nu(x, diam)
  assert r - N < tol, "r = " + str(r) + "N = " + str(N) #only check if r < N here
  
def test_ball():
  for D in Ds:
    yield check_ball_data, 10*D, D

def check_affine_data(D, Da):
  N = 2**Da
  x = np.zeros((N, Da))
  for i in range(N):
    for d in range(Da):
      x[i, d] = (i >> d) % 2
  x[x == 0] = -1
  xfix = (np.random.rand(D-Da) > 0.5).astype(int)
  xfix[xfix == 0] = -1
  x = np.hstack((x, xfix*np.ones((N, D-Da))))

  #verify diam/normratio for untransformed data
  diam = compute_diam(x)
  assert np.fabs(diam -  np.sqrt(float(Da)/float(D))*2.) < tol, "diam = " + str(diam)

  normratio = compute_normratio(x)
  assert np.fabs(normratio - np.sqrt(float(Da)/float(D))) < tol, "nrmratio = " + str(normratio)

  #do a unitary + translation
  U = np.random.normal(0., 1., (D, D))
  U = U.T.dot(U)
  _, W = np.linalg.eigh(U)
  t = np.random.normal(0., 1., D)
  xP = x.dot(W) + t
  
  #recompute diam + radius and check
  diam = compute_diam(x)
  nu, r = compute_nu(x, diam)
  assert np.fabs(r - N) < tol, "r = " + str(r) + "N = " + str(N)
  
def test_affine():
  for D in range(2, 10):
    for Da in range(1, D): 
      yield check_affine_data, D, Da

def check_single_data(D):
  x = np.random.normal(0., 1., D)
  x /= np.sqrt((x**2).sum())
  
  diam = compute_diam(np.atleast_2d(x))
  assert np.fabs(diam - 0.) < tol, "diam = " + str(diam)

  normratio = compute_normratio(np.atleast_2d(x))
  assert np.fabs(normratio - 0.) < tol, "nrmratio = " + str(normratio)

  nu, r = compute_nu(np.atleast_2d(x), diam)
  assert np.fabs(r - 0.) < tol, "r = " + str(r) + " (should be 0)"

def test_single_data():
  for D in Ds:
    yield check_single_data, D

  
  
