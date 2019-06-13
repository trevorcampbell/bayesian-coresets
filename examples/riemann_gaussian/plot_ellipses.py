import bokeh.layouts as bkl
from utils import *
import bokeh.plotting as bkp
import numpy as np

size_x_axis = True
scaled = True
trial_num = 0
nm = ('RAND', 'Uniform')
Ms = [0, 1, 2, 5, 10, 20, 50, 99]


#plot the sequence of coreset pts and comparison of nonopt + opt
res = np.load('results/results_'+nm[0] + '_' + str(trial_num)+'.npz')
x = res['x']
wt = res['w']
wt_opt = res['w_opt']
Sig = res['Sig']
mup = res['mup']
Sigp = res['Sigp']
muwt = res['muw']
Sigwt = res['Sigw']
muwt_opt = res['muw_opt']
Sigwt_opt = res['Sigw_opt']

nms = [('SVI1', 'SparseVI-1'), ('SVIF', 'SparseVI-Full'), ('GIGAT', 'GIGA (Truth)'), ('GIGAN', 'GIGA (Noisy)'), ('RAND', 'Uniform')]


#plot the example set of 3-sigma ellipses
figs = []
for m in Ms:
  fig = bkp.figure(x_range=(-5,5), y_range=(-5,5), plot_width=750, plot_height=750)
  #plot the data
  x = np.load('results/results_'+nm[0] + '_' + str(trial_num)+'.npz')['x']
  fig.scatter(x[:, 0], x[:, 1], fill_color='black', alpha=0.1)
  for i, nm in enumerate(nms):
    res = np.load('results/results_'+nm[0]+'_'+('scaled' if scaled else 'unscaled') + '_' + str(trial_num)+'.npz')
    wt = res['w']
    wt_opt = res['w_opt']
    Sig = res['Sig']
    mup = res['mup']
    Sigp = res['Sigp']
    muwt = res['muw']
    Sigwt = res['Sigw']
    muwt_opt = res['muw_opt']
    Sigwt_opt = res['Sigw_opt']
    rklw = res['rklw']
    fklw = res['fklw']
    rklw_opt = res['rklw_opt']
    fklw_opt = res['fklw_opt']
    plot_gaussian(fig, mup, Sigp, Sig, 'black', 5, 3, 1, 1, 'solid', 'True')
    plot_gaussian(fig, muwt[m,:], Sigwt[m,:,:], Sig, pal[i], 5, 3, 1, 1, 'dashed', nm[1])
    plot_gaussian(fig, muwt_opt[m,:], Sigwt_opt[m,:], Sig, pal[i], 5, 3, 1, 1, 'solid', nm[1])
  figs.append(fig)

bkp.show(bkl.gridplot(figs))
