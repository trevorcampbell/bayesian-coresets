import bokeh.plotting as bkp
import numpy as np
import sys, os
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *


plot_reverse_kl = True
trials = np.arange(1, 11)
#nms = [('SVI1', 'SparseVI-1'), ('SVIF', 'SparseVI-Full'), ('GIGAT', 'GIGA (Truth)'), ('GIGAN', 'GIGA (Noisy)'), ('IH', 'HOPS'), ('RAND', 'Uniform')]
nms = [('SVIF', 'SparseVI'), ('GIGAT', 'GIGA (Truth)'), ('GIGAN', 'GIGA (Noisy)'), ('RAND', 'Uniform')]


#plot the KL figure
fig = bkp.figure(y_axis_type='log', plot_width=750, plot_height=750, x_axis_label='Coreset Size', y_axis_label=('Reverse KL' if plot_reverse_kl else 'Forward KL') )
preprocess_plot(fig, '32pt', False, True)

plot_every = 5

for i, nm in enumerate(nms):
  kl = []
  sz = []
  for t in trials:
    res = np.load('results/results_'+nm[0]+'_' + str(t)+'.npz')
    if plot_reverse_kl:
      kl.append(res['rklw'][::plot_every])
    else:
      kl.append(res['fklw'][::plot_every])
    sz.append((res['w'] > 0).sum(axis=1)[::plot_every])
  fig.circle(np.percentile(sz, 50, axis=0), np.percentile(kl, 50, axis=0), color=pal[i], line_width=5, legend=nm[1]) 
  fig.segment(x0 = np.percentile(sz, 50, axis=0), x1 = np.percentile(sz, 50, axis=0), y0 = np.percentile(kl, 25, axis=0), y1 = np.percentile(kl, 75, axis=0), color=pal[i], line_width=5, legend=nm[1]) 
  fig.segment(x0 = np.percentile(sz, 25, axis=0), x1 = np.percentile(sz, 75, axis=0), y0 = np.percentile(kl, 50, axis=0), y1 = np.percentile(kl, 50, axis=0), color=pal[i], line_width=5, legend=nm[1]) 

postprocess_plot(fig, '22pt', location='bottom_left', glyph_width=40)
fig.legend.background_fill_alpha=0.
fig.legend.border_line_alpha=0.

bkp.show(fig)



