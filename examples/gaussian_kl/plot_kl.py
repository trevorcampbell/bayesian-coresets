from utils import *
import bokeh.plotting as bkp
import numpy as np

plot_reverse_kl = True
size_x_axis = True
scaled = True
n_trials = 10
nms = [('EGUS', 'Uniform'), ('ERG', 'Greedy'), ('ERL1', 'L1')]
Ms = np.arange(100)

#plot the KL figure
fig = bkp.figure(y_axis_type='log', plot_width=750, plot_height=750, x_axis_label=('Coreset Size' if size_x_axis else '# Iterations'), y_axis_label=('Reverse KL' if plot_reverse_kl else 'Forward KL') )
preprocess_plot(fig, '24pt', True)

for i, nm in enumerate(nms):
  kl = []
  klopt = []
  for t in range(n_trials):
    res = np.load('results_'+nm[0]+'_' + str(t)+'.npz')
    if plot_reverse_kl:
      klt = res['rklw']
      kloptt = res['rklw_opt']
    else:
      klt = res['fklw']
      kloptt = res['fklw_opt']
    sz = (res['w'] > 0).sum(axis=1) 
    szopt = (res['w_opt'] > 0).sum(axis=1) 
    if size_x_axis:
      kl.append(np.interp(Ms, sz, klt))
      klopt.append(np.interp(Ms, szopt, kloptt))
    else:
      kl.append(klt)
      klopt.append(kloptt)
    #fig.scatter(sz[-1], kl[-1], color=pal[i], legend=nm) 
    #fig.scatter(szopt[-1], klopt[-1], color=pal[i], legend=nm) 
  if size_x_axis:
    fig.line(Ms, np.percentile(kl, 50, axis=0), color=pal[i], line_width=5, line_dash='dashed', legend=nm[1]) 
    fig.line(Ms, np.percentile(klopt, 50, axis=0), color=pal[i], line_width=5, line_dash='solid', legend=nm[1]) 
  else:
    kl = np.array(kl)
    klopt = np.array(klopt)
    fig.line(np.arange(kl.shape[1]), np.percentile(kl, 50, axis=0), color=pal[i], line_width=5, line_dash='dashed', legend=nm[1])
    fig.line(np.arange(kl.shape[1]), np.percentile(klopt, 50, axis=0), color=pal[i], line_width=5, line_dash='solid', legend=nm[1])
    #plot_meanstd(fig, np.arange(kl.shape[1]), klopt, pal[i], 5, 0.3, 'solid', nm)

postprocess_plot(fig, '16pt')

bkp.show(fig)



