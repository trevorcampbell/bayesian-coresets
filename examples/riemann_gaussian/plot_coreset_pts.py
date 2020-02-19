import bokeh.layouts as bkl
import pickle as pk
import bokeh.plotting as bkp
import numpy as np
import sys,os
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *


np.random.seed(5)
trial_num = 1
nm = ('SVI', 'SparseVI')
#Ms = [0, 1, 2, 5, 8, 12, 20, 50, 100, 150, 200]
Ms = [0, 1, 2]

#plot the KL figure

#plot the sequence of coreset pts and comparison of nonopt + opt
f = open('results/results_'+nm[0]+'_' +str(trial_num)+'.pk', 'rb')
res = pk.load(f)#res = (x, mu0, Sig0, Sig, mup, Sigp, w, p, muw, Sigw, rklw, fklw)
f.close()
x = res[0]
wt = res[6]
pt = res[7]
Sig = res[3]
mup = res[4]
Sigp = res[5]
muwt = res[8]
Sigwt = res[9]

#if dim x > 2, project onto two random orthogonal axes
if x.shape[1] > 2:
  ##centering on the true data gen dist
  #true_th = np.ones(mup.shape[0])
  #x-= true_th
  #mup -= true_th
  #muwt -= true_th
  #project onto two random axes
  a1 = np.random.randn(x.shape[1])
  a2 = np.random.randn(x.shape[1])
  a1 /= np.sqrt((a1**2).sum())
  a2 -= a2.dot(a1)*a1
  a2 /= np.sqrt((a2**2).sum())
  a = np.hstack((a1[:,np.newaxis], a2[:,np.newaxis]))
  x = x.dot(a)
  mup = mup.dot(a)
  muwt = muwt.dot(a)
  Sig = a.T.dot(Sig.dot(a))
  Sigp = a.T.dot(Sigp.dot(a))
  Sigwttmp = np.zeros((Sigwt.shape[0], 2, 2))
  for i in range(Sigwt.shape[0]):
    Sigwttmp[i,:,:] = a.T.dot(Sigwt[i,:,:].dot(a))
  Sigwt = Sigwttmp
  pt = [p.dot(a) for p in pt]
  ##shift everything to be back to true th
  #true_th = true_th[:2]
  #x += true_th
  #mup += true_th
  #muwt += true_th

figs = []
for m in Ms:
  x_range = (-4.2, 4.2)
  y_range = (-3, 5.4)
  fig = bkp.figure(x_range=x_range, y_range=y_range, plot_width=750, plot_height=750)
  preprocess_plot(fig, '24pt', False, False)

  fig.scatter(x[:, 0], x[:, 1], fill_color='black', size=10, alpha=0.09)

  fig.scatter(pt[m][:, 0], pt[m][:, 1], fill_color='black', size=10+40*wt[m]/wt[m].max(), line_color=None)

  plot_gaussian(fig, mup, (4./9.)*Sigp, (4./9.)*Sig, 'black', 17, 9, 1, 1, 'solid', 'Exact')

  plot_gaussian(fig, muwt[m,:], (4./9.)*Sigwt[m,:], (4./9.)*Sig, pal[0], 17, 9, 1, 1, 'solid', nm[1]+', ' + str(m) +' pts') 

  postprocess_plot(fig, '24pt', orientation='horizontal', glyph_width=80)
  fig.legend.background_fill_alpha=0.
  fig.legend.border_line_alpha=0.
  #f.legend.visible=False
  fig.xaxis.visible = False
  fig.yaxis.visible = False

  figs.append(fig)

bkp.show(bkl.gridplot([figs]))




