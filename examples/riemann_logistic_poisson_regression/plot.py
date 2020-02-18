import numpy as np
import pickle as pk
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import time
import os, sys
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *


dnames = ['synth_lr', 'ds1', 'phishing', 'synth_poiss', 'biketrips', 'airportdelays']
algs = [('RAND', 'Uniform', pal[3]),  ('SVI', 'SparseVI', pal[0]), ('BPSVI', 'BPSVI', pal[4]), ('GIGAR','GIGA (Realistic)', pal[2]),('GIGAO','GIGA (Optimal)', pal[1]), ('PRIOR','Prior', 'black')]

figs = []
for dnm in dnames:
  print('Plotting ' + dnm)
  fig = bkp.figure(y_axis_type='log', y_axis_label='Reverse KL',  x_axis_label='# Iterations', width=2400, height=800)
  preprocess_plot(fig, '72pt', False, True)
  fig2 = bkp.figure(y_axis_type='log', y_axis_label='Reverse KL', x_axis_type='log', x_axis_label='CPU Time (s)', width=2400, height=800)
  preprocess_plot(fig2, '72pt', True, True)
  fig3 = bkp.figure(y_axis_type='log', y_axis_label='Reverse KL',  x_axis_label='Coreset Size', width=2400, height=800)
  preprocess_plot(fig3, '72pt', False, True)

  figs.append([fig, fig2, fig3])
  
  #get normalizations based on the prior
  std_kls = {}
  trials = [fn for fn in os.listdir('results/') if dnm+'_PRIOR_results_' in fn]
  if len(trials) == 0: 
    print('Need to run prior to establish baseline first')
    quit()
  kltot = 0.
  M = 0
  for tridx, fn in enumerate(trials):
    f = open('results/'+fn, 'rb')
    res = pk.load(f) #(cputs, w, p, mus_laplace, Sigs_laplace, rkls_laplace, fkls_laplace)
    f.close()
    assert np.all(res[5] == res[5][0]) #make sure prior doesn't change...
    kltot += res[5][0]
    M = res[0].shape[0]
  std_kls[dnm] = kltot / len(trials)
  
  for alg in algs:
    trials = [fn for fn in os.listdir('results/') if dnm+'_'+alg[0]+'_results_' in fn]
    if len(trials) == 0: continue
    kls = np.zeros((len(trials), M))
    cputs = np.zeros((len(trials), M))
    cszs = np.zeros((len(trials), M))
    kl0 = std_kls[dnm] 
    for tridx, fn in enumerate(trials):
      f = open('results/'+fn, 'rb')
      res = pk.load(f) #(cputs, w, p, mus_laplace, Sigs_laplace, rkls_laplace, fkls_laplace)
      f.close()
      cputs[tridx, :] = res[0]
      wts = res[1]
      mu = res[3]
      Sig = res[4]
      kl = res[5]
      cszs[tridx, :] = np.array([w.shape[0] for w in wts])
      kls[tridx, :] = kl/kl0
      if 'PRIOR' in fn:
        kls[tridx, :] = np.median(kls[tridx,:])

    # since cputs record per-iteration time, need to sum up for everything except BPSVI
    if alg[0] != 'BPSVI':
      cputs = np.cumsum(cputs, axis=1)
  
    cput50 = np.percentile(cputs, 50, axis=0)
    cput25 = np.percentile(cputs, 25, axis=0)
    cput75 = np.percentile(cputs, 75, axis=0)
  
    csz50 = np.percentile(cszs, 50, axis=0)
    csz25 = np.percentile(cszs, 25, axis=0)
    csz75 = np.percentile(cszs, 75, axis=0)
  
    kl50 = np.percentile(kls, 50, axis=0)
    kl25 = np.percentile(kls, 25, axis=0)
    kl75 = np.percentile(kls, 75, axis=0)
  
    fig.line(np.arange(kl50.shape[0]), kl50, color=alg[2], legend=alg[1], line_width=10)
    fig.line(np.arange(kl25.shape[0]), kl25, color=alg[2], legend=alg[1], line_width=10, line_dash='dashed')
    fig.line(np.arange(kl75.shape[0]), kl75, color=alg[2], legend=alg[1], line_width=10, line_dash='dashed')
  
    fig2.line(cput50, kl50, color=alg[2], legend=alg[1], line_width=10)
    fig2.patch(np.hstack((cput50, cput50[::-1])), np.hstack((kl75, kl25[::-1])), fill_color=alg[2], legend=alg[1], alpha=0.3)
  
    #fig2.circle(cput50, kl50, color=alg[2], legend=alg[1], size=15)
    #fig2.segment(x0=cput25, x1 = cput75, y0 = kl50, y1 = kl50, color=alg[2], legend=alg[1], line_width=4)
    #fig2.segment(y0=kl25, y1 = kl75, x0 = cput50, x1 = cput50, color=alg[2], legend=alg[1], line_width=4)

  
    if dnm != 'PRIOR':
      fig3.line(csz50, kl50, color=alg[2], legend=alg[1], line_width=10)
      fig3.patch(np.hstack((csz50, csz50[::-1])), np.hstack((kl75, kl25[::-1])), fill_color=alg[2], legend=alg[1], alpha=0.3)

      #fig3.circle(csz50, kl50, color=alg[2], legend=alg[1], size=15)
      #fig3.segment(x0=csz25, x1 = csz75, y0 = kl50, y1 = kl50, color=alg[2], legend=alg[1], line_width=4)
      #fig3.segment(y0=kl25, y1 = kl75, x0 = csz50, x1 = csz50, color=alg[2], legend=alg[1], line_width=4)
  
     
  for f in [fig, fig2, fig3]:
    f.legend.label_text_font_size= '72pt'
    f.legend.glyph_width=80
    f.legend.glyph_height=80
    f.legend.spacing=20
    f.legend.visible = False

bkp.show(bkl.gridplot(figs))
