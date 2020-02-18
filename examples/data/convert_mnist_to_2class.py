import numpy as np
with np.load('mnist.npz') as data:
  X=data['X']
  y=data['y']
  print(data['y'])
  y=np.where(y==2, 1, -1)
  np.savez('mnist2class', X=X, y=y)
