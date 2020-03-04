import bokeh.plotting as bkp
from bokeh.io import export_png, export_svgs
import numpy as np
import sys, os
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../bayesian-coresets-private/examples/common'))
from plotting import *

n_trials=int(sys.argv[1])
plot_every=int(sys.argv[2])
d=sys.argv[3]
N=sys.argv[4]
fldr=sys.argv[5]
prfx=sys.argv[6]
no_pcst=sys.argv[7]

plot_reverse_kl = True
trials = np.arange(1, n_trials)
nms = [('BPSVI', 'PSVI'), ('SVI', 'SparseVI'),   ('RAND', 'Uniform'), ('GIGAO', 'GIGA (Optimal)'), ('GIGAR', 'GIGA (Realistic)')]
pcsts = ['PSVI'] # pseudocoreset nameslist

#plot the KL figure
fig = bkp.figure(y_axis_type='log', plot_width=850, plot_height=850, x_axis_label='Coreset Size',
       y_axis_label=('Reverse KL' if plot_reverse_kl else 'Forward KL'), toolbar_location=None )
preprocess_plot(fig, '32pt', False, True)

for i, nm in enumerate(nms):
  kl = []
  sz = []
  for tr in trials:
    numTuple = (prfx, nm[0], str(d), str(tr))
    x_, mu0_, Sig0_, Sig_, mup_, Sigp_, w_, p_, muw_, Sigw_, rklw_, fklw_ = np.load(os.path.join(fldr, '_'.join(numTuple)+'.pk'), allow_pickle=True)
    if plot_reverse_kl:
      kl.append(rklw_[::plot_every])
    else:
      kl.append(fklw_[::plot_every])
    if nm[0] in pcsts:
      sz.append(range(len(w_))[::plot_every])
    else:
      sz.append([np.count_nonzero(a) for a in w_[::plot_every]])
  x = np.percentile(sz, 50, axis=0)
  # HACK FOR REPRODUCING COLOURS OF NEURIPS PAPER:
  #assign PSVI to last colour of the pallete in order to maintain colouring of previous papers for baselines
  if no_pcst=="True":
    fig.line(x, np.percentile(kl, 50, axis=0), color=pal[i], line_width=5, legend=nm[1])
    fig.patch(x = np.hstack((x, x[::-1])), y = np.hstack((np.percentile(kl, 75, axis=0), np.percentile(kl, 25, axis=0)[::-1])), color=pal[i], fill_alpha=0.4, legend=nm[1])
  else:
    fig.line(x, np.percentile(kl, 50, axis=0), color=pal[i-1], line_width=5, legend=nm[1])
    fig.patch(x = np.hstack((x, x[::-1])), y = np.hstack((np.percentile(kl, 75, axis=0), np.percentile(kl, 25, axis=0)[::-1])), color=pal[i-1], fill_alpha=0.4, legend=nm[1])

postprocess_plot(fig, '20pt', location='bottom_right', glyph_width=40)
fig.legend.background_fill_alpha=0.
fig.legend.border_line_alpha=0.
fig.legend.visible = False

bkp.show(fig)
#fig.output_backend = "svg"
fldr_figs = 'figs'
if not os.path.exists(fldr_figs):
  os.mkdir(fldr_figs)
export_png(fig, filename=os.path.join(fldr_figs, "d"+str(d)+"_KLDvsCstSize.png"), height=1500, width=1500)
