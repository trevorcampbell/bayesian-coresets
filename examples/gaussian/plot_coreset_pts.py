import bokeh.layouts as bkl
import bokeh.plotting as bkp
from bokeh.io import export_png, export_svgs
import numpy as np
import sys,os
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../bayesian-coresets-private/examples/common'))
from plotting import *

size_x_axis = True
trial_num = 1
np.random.seed(42)
nms = [('BPSVI', 'PSVI', np.load('results/results_BPSVI_200_0.pk', allow_pickle=True)), ('SVI', 'SparseVI', np.load('results/results_SVI_200_0.pk', allow_pickle=True))]
Ms = [0, 1, 2, 5, 8, 12, 20, 30, 50, 100, 150, 200]
d=200

figs = []
sizebase=20
alphagauss=0.6

#project onto two random axes
a1 = np.random.randn(d)
a2 = np.random.randn(d)
a1 /= np.sqrt((a1**2).sum())
a2 -= a2.dot(a1)*a1
a2 /= np.sqrt((a2**2).sum())
a = np.hstack((a1[:,np.newaxis], a2[:,np.newaxis]))

fldr_figs = 'figs'
if not os.path.exists(fldr_figs):
  os.mkdir(fldr_figs)
for m in Ms:
  u = np.zeros((m+1, m, d))
  w = np.zeros((m+1, m))
  res=nms[0][2]
  x_, mu0_, Sig0_, Sig_, mup_, Sigp_, w_, p_, muw_, Sigw_, rklw_, fklw_ = res
  # turn pseudodata points into 3d array to enable einsum
  i=0
  while(i<=m):
    ms = p_[i].shape[0]
    u[i, :ms, :] = p_[i]
    i+=1
  x_range = (-4.2, 4.2)
  y_range = (-3.5, 4.9)
  fig = bkp.figure(x_range=x_range, y_range=y_range, plot_width=750, plot_height=750, toolbar_location=None)
  preprocess_plot(fig, '24pt', False, False)
  if d>2: # fix this to make it independent of ordering of input
    # psvi
    p_ = np.einsum('ijk,kl->ijl', u, a)
    mup_ = mup_.dot(a)
    Sig_ = a.T.dot(Sig_.dot(a))
    Sigp_ = a.T.dot(Sigp_.dot(a))
    muw_ = muw_.dot(a)
    Sigwttmp = np.zeros((Sigw_.shape[0], 2, 2))
    for i in range(Sigw_.shape[0]):
      Sigwttmp[i,:,:] = a.T.dot(Sigw_[i,:,:].dot(a))
    Sigw_ = Sigwttmp
    msz = np.where((w_[m] > 0))
    # check if rows of x do correspond to optimized pseudopoints or
    # just dummy initialization to [0., 0.], in which case get filtered out
    um = p_[m, :, :]
    um = um[np.abs(np.sum(um, axis=1)) > .00000000001]

    # svi
    svires=nms[1][2]
    svix_, svimu0_, sviSig0_, sviSig_, svimup_, sviSigp_, sviw_, svip_, svimuw_, sviSigw_, svirklw_, svifklw_ = svires
    svix_ = svix_.dot(a)
    svimuw_ = svimuw_.dot(a)
    sviSigwttmp = np.zeros((sviSigw_.shape[0], 2, 2))
    for i in range(sviSigw_.shape[0]):
      sviSigwttmp[i,:,:] = a.T.dot(sviSigw_[i,:,:].dot(a))
    sviSigw_ = sviSigwttmp
    svimsz = np.where((sviw_[m] > 0))
    # check if rows of x do correspond to optimized pseudopoints or
    # just dummy initialization to [0., 0.], in which case get filtered out
    svix = svix_[np.abs(np.sum(svix_, axis=1)) > .00000000001]

    # produce figs
    fig.scatter(svix[:, 0], svix[:, 1], fill_color='black', size=sizebase, alpha=0.15)
    plot_gaussian(fig, mup_, (4./9.)*Sigp_, (4./9.)*Sig_, 'black', 17, 9, 1, 1, 'solid', 'Exact')
    if size_x_axis:
      fig.scatter(um[:, 0], um[:, 1], fill_color='maroon', alpha=0.6, size=sizebase*(w_[m]>0)+4*sizebase*w_[m]/w_[m].max(), line_color=None)
    else:
      fig.scatter(um[:, 0], um[:, 1], fill_color='maroon', alpha=0.6, size=sizebase*(w_[msz, :]>0)+4*sizebase*w_[m,:]/w_[m,:].max(), line_color=None)
    if size_x_axis:
      if m==0:
        plot_gaussian(fig, muw_[m]+0.02, (4./9.)*Sigw_[m], (4./9.)*Sig_, 'maroon', 17, 9, 1, alphagauss, 'solid',  nms[0][1] )
      else:
        plot_gaussian(fig, muw_[m], (4./9.)*Sigw_[m], (4./9.)*Sig_, 'maroon', 17, 9, 1, alphagauss, 'solid',  nms[0][1] )
    else:
      if m==0:
        plot_gaussian(fig, muw_[m,:]+0.02, (4./9.)*Sigw_[m,:], (4./9.)*Sig_, 'maroon', 17, 9, 1, alphagauss, 'solid',   nms[0][1])
      else:
        plot_gaussian(fig, muw_[m,:], (4./9.)*Sigw_[m,:], (4./9.)*Sig_, 'maroon', 17, 9, 1, alphagauss, 'solid',   nms[0][1])
    if size_x_axis:
      fig.scatter(svix_[:, 0], svix_[:, 1], fill_color='blue', alpha=0.6, size=sizebase*(sviw_[m]>0)+4*sizebase*sviw_[m]/sviw_[m].max(), line_color=None)
    else:
      fig.scatter(svix_[:, 0], svix_[:, 1], fill_color='blue', alpha=0.6, size=sizebase*(sviw_[svimsz, :]>0)+4*sizebase*sviw_[m,:]/sviw_[m,:].max(), line_color=None)
    if size_x_axis:
      plot_gaussian(fig, svimuw_[m,:], (4./9.)*sviSigw_[m,:], (4./9.)*Sig_, pal[0], 17, 9, 1, alphagauss, 'solid', nms[1][1])
    else:
      plot_gaussian(fig, svimuw_[m,:], (4./9.)*sviSigw_[m,:], (4./9.)*Sig_, pal[0], 17, 9, 1, alphagauss, 'solid', nms[1][1])
    postprocess_plot(fig, '36pt', orientation='horizontal', glyph_width=80)
    fig.legend.background_fill_alpha=0.
    fig.legend.border_line_alpha=0.
    fig.legend.visible=(m==Ms[-1])
    fig.xaxis.visible = False
    fig.yaxis.visible = False
    figs.append(fig)
    export_png(fig, filename=os.path.join(fldr_figs, "d"+str(d)+"_pts"+str(m)+".png"), height=1500, width=1500)
#export_png(fig, filename=os.path.join(fldr_figs, "d"+str(d)+"_pts.png"),height=1500, width=1500)
