import numpy as np
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import time
import os, sys
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *


#algs = [('RAND', 'Uniform', pal[7]),  ('SVIF', 'SparseVI', pal[2]), ('QSVIF', 'Quadratic SparseVI', pal[4]), ('GIGAN','GIGA (Noisy)', pal[1]),('GIGAT','GIGA (Truth)', pal[0]), ('PRIOR','Prior', 'black')]
dnames = ['synth_lr', 'ds1', 'phishing', 'synth_poiss', 'biketrips', 'airportdelays']
algs = [('RAND', 'Uniform', pal[3]),  ('SVIF', 'SparseVI', pal[0]), ('GIGAN','GIGA (Noisy)', pal[2]),('GIGAT','GIGA (Truth)', pal[1]), ('PRIOR','Prior', 'black')]

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
  for tridx, fn in enumerate(trials):
    res = np.load('results/'+fn)
    assert np.all(res['kls'] == res['kls'][0]) #make sure prior doesn't change...
    kltot += res['kls'][0]
  std_kls[dnm] = kltot / len(trials)
  
  for alg in algs:
    trials = [fn for fn in os.listdir('results/') if dnm+'_'+alg[0]+'_results_' in fn]
    if len(trials) == 0: continue
    Ms = np.load('results/'+trials[0])['Ms']
    kls = np.zeros((len(trials), len(Ms)))
    cputs = np.zeros((len(trials), len(Ms)))
    cszs = np.zeros((len(trials), len(Ms)))
    kl0 = std_kls[dnm] 
    for tridx, fn in enumerate(trials):
      res = np.load('results/'+fn)
      cput = res['cputs']
      cputs[tridx, :] = cput[:len(Ms)]
      wts = res['wts']
      mu = res['mus']
      Sig = res['Sigs']
      kl = res['kls']
      cszs[tridx, :] = (wts > 0).sum(axis=1)
      kls[tridx, :] = kl[:len(Ms)]/kl0
      if 'PRIOR' in fn:
        kls[tridx, :] = np.median(kls[tridx,:])
  
    cput50 = np.percentile(cputs, 50, axis=0)
    cput25 = np.percentile(cputs, 25, axis=0)
    cput75 = np.percentile(cputs, 75, axis=0)
  
    csz50 = np.percentile(cszs, 50, axis=0)
    csz25 = np.percentile(cszs, 25, axis=0)
    csz75 = np.percentile(cszs, 75, axis=0)
  
    kl50 = np.percentile(kls, 50, axis=0)
    kl25 = np.percentile(kls, 25, axis=0)
    kl75 = np.percentile(kls, 75, axis=0)
  
    fig.line(Ms, kl50, color=alg[2], legend=alg[1], line_width=10)
    fig.line(Ms, kl25, color=alg[2], legend=alg[1], line_width=10, line_dash='dashed')
    fig.line(Ms, kl75, color=alg[2], legend=alg[1], line_width=10, line_dash='dashed')
  
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
