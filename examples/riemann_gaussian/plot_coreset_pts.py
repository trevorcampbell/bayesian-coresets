import bokeh.layouts as bkl
import bokeh.plotting as bkp
import numpy as np
import sys,os
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *


size_x_axis = False
trial_num = 1

nm = ('SVIF', 'SparseVI')
Ms = [0, 1, 2, 5, 8, 12, 20, 50, 100, 150, 200]

np.random.seed(5)
#plot the KL figure

#plot the sequence of coreset pts and comparison of nonopt + opt
res = np.load('results/results_'+nm[0] + '_' + str(trial_num)+'.npz')
x = res['x']
wt = res['w']
Sig = res['Sig']
mup = res['mup']
Sigp = res['Sigp']
muwt = res['muw']
Sigwt = res['Sigw']

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

  msz = np.where((wt > 0).sum(axis=1) <= m)[0][-1]
  fig.scatter(x[:, 0], x[:, 1], fill_color='black', size=10, alpha=0.09)

  if size_x_axis:
    fig.scatter(x[:, 0], x[:, 1], fill_color='black', size=10*(wt[msz, :]>0)+40*wt[msz,:]/wt[msz,:].max(), line_color=None)
  else:
    fig.scatter(x[:, 0], x[:, 1], fill_color='black', size=10*(wt[msz, :]>0)+40*wt[m,:]/wt[m,:].max(), line_color=None)

  plot_gaussian(fig, mup, (4./9.)*Sigp, (4./9.)*Sig, 'black', 17, 9, 1, 1, 'solid', 'Exact')

  if size_x_axis:
    plot_gaussian(fig, muwt[msz,:], (4./9.)*Sigwt[msz,:], (4./9.)*Sig, pal[0], 17, 9, 1, 1, 'solid', nm[1]+', size ' + str( (wt[msz, :]>0).sum() ))
  else:
    plot_gaussian(fig, muwt[m,:], (4./9.)*Sigwt[m,:], (4./9.)*Sig, pal[0], 17, 9, 1, 1, 'solid', nm[1]+', ' + str(m) +' pts') 

  postprocess_plot(fig, '24pt', orientation='horizontal', glyph_width=80)
  fig.legend.background_fill_alpha=0.
  fig.legend.border_line_alpha=0.
  #f.legend.visible=False
  fig.xaxis.visible = False
  fig.yaxis.visible = False

  figs.append(fig)

bkp.show(bkl.gridplot([figs]))




