import numpy as np
from hilbertcoresets.geometry import *

tol = 1e-9
n_trials = 50

def check_box_data(D):
  N = 2**D
  x = np.zeros((N, D))
  for i in range(N):
    for d in range(D):
      x[i, d] = (i >> d) % 2
  x[x == 0] = -1

  diam = compute_diam(x)
  assert np.fabs(diam -  2.) < tol

  normratio = compute_normratio(x)
  assert np.fabs(normratio - 1.) < tol

  nu, r = compute_nu(x, diam)
  assert np.fabs(r - N) < tol

def test_box():
  for D in range(1, 10):
    yield check_box_data, D


def check_ball_data(N, D):
  x = np.random.normal(0., 1., (N/2, D))
  x /= np.sqrt((x**2).sum(axis=1))[:, np.newaxis]
  x = np.vstack((x, -x))
  N = x.shape[0]
  
  diam = compute_diam(x)
  assert np.fabs(diam -  2.) < tol

  normratio = compute_normratio(x)
  assert np.fabs(normratio - 1.) < tol

  nu, r = compute_nu(x, diam)
  assert np.fabs(r - N) < tol
  
def test_ball():
  for D in range(1, 10):
    yield check_ball_data, 10*D, D

#def check_affine_data(N, D, Da):
#  x = np.random.normal(0., 1., (N, D))
#  t = np.random.normal(0., 1., D)
#  mx = x.mean(axis=0)
#  x -= mx
#  cov = x.T.dot(x)
#  w, W = np.linalg.eigh(cov)
#  W = W[:, :Da]
#  xP = x.dot(W) + t
#  
#def test_affine():
#  for D in range(2, 10):
#    for Da in range(1, D): 
#      yield check_affine_data, 10*D, D, Da

def check_single_data(D):
  x = np.random.normal(0., 1., D)
  x /= np.sqrt((x**2).sum())
  
  diam = compute_diam(np.atleast_2d(x))
  assert np.fabs(diam - 0.) < tol

  normratio = compute_normratio(np.atleast_2d(x))
  assert np.fabs(normratio - 1.) < tol

  nu, r = compute_nu(np.atleast_2d(x), diam)
  assert np.fabs(r - N) < tol

def test_single_data():
  for D in range(1, 10):
    yield check_single_data, D

  
  
