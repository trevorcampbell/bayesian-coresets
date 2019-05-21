import bokeh.layouts as bkl
from utils import *
import bokeh.plotting as bkp
import numpy as np

size_x_axis = False
trial_num = 0
nm = ('EGUS', 'Uniform')
Ms = [0, 1, 2, 5, 10, 20, 50, 99]

nm = ('ERL1', 'L1')
Ms = np.arange(14)

#nm = ('ERL1U', 'L1U')
#Ms = np.arange(20)
nm = ('ERG', 'Greedy')
Ms = np.arange(14)

nm = ('ERCG', 'Fully Corrective Greedy')
Ms = np.arange(20)

Ms = [0, 1, 2, 5, 8, 12]


np.random.seed(5)
#plot the KL figure

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

#if dim x > 2, project onto two random orthogonal axes
if x.shape[1] > 2:
  ##centering on the true data gen dist
  #true_th = np.ones(mup.shape[0])
  #x-= true_th
  #mup -= true_th
  #muwt -= true_th
  #muwt_opt -= true_th
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
  muwt_opt = muwt_opt.dot(a)
  Sig = a.T.dot(Sig.dot(a))
  Sigp = a.T.dot(Sigp.dot(a))
  Sigwttmp = np.zeros((Sigwt.shape[0], 2, 2))
  Sigwtopttmp = np.zeros((Sigwt.shape[0], 2, 2))
  for i in range(Sigwt.shape[0]):
    Sigwttmp[i,:,:] = a.T.dot(Sigwt[i,:,:].dot(a))
    Sigwtopttmp[i,:,:] = a.T.dot(Sigwt_opt[i,:,:].dot(a))
  Sigwt = Sigwttmp
  Sigwt_opt = Sigwtopttmp
  ##shift everything to be back to true th
  #true_th = true_th[:2]
  #x += true_th
  #mup += true_th
  #muwt += true_th
  #muwt_opt += true_th

figs = []
for m in Ms:
  x_range = (-4.2, 4.2)
  y_range = (-3, 5.4)
  fig = bkp.figure(x_range=x_range, y_range=y_range, plot_width=750, plot_height=750)
  fig_opt = bkp.figure(x_range=x_range, y_range=y_range, plot_width=750, plot_height=750)
  #for f in [fig, fig_opt]:
  for f in [fig]:
    preprocess_plot(f, '24pt', False)

  #for (f, w, muw, Sigw) in [(fig, wt, muwt, Sigwt), (fig_opt, wt_opt, muwt_opt, Sigwt_opt)]:
  for (f, w, muw, Sigw) in [(fig, wt, muwt, Sigwt)]:
    msz = np.where((w > 0).sum(axis=1) <= m)[0][-1]
    f.scatter(x[:, 0], x[:, 1], fill_color='black', size=10, alpha=0.09)

    if size_x_axis:
      f.scatter(x[:, 0], x[:, 1], fill_color='black', size=10*(w[msz, :]>0)+40*w[msz,:]/w[msz,:].max())
    else:
      f.scatter(x[:, 0], x[:, 1], fill_color='black', size=10*(w[msz, :]>0)+40*w[m,:]/w[m,:].max())

    plot_gaussian(f, mup, (4./9.)*Sigp, (4./9.)*Sig, pal[0], 17, 9, 1, 1, 'solid', 'Exact')

    if size_x_axis:
      plot_gaussian(f, muw[msz,:], (4./9.)*Sigw[msz,:], (4./9.)*Sig, pal[2], 17, 9, 1, 1, 'solid', nm[1]+', size ' + str( (w[msz, :]>0).sum() ))
    else:
      plot_gaussian(f, muw[m,:], (4./9.)*Sigw[m,:], (4./9.)*Sig, pal[2], 17, 9, 1, 1, 'solid', nm[1]+', ' + str(m) +' pts') 


  #for f in [fig, fig_opt]:
  for f in [fig]:
    postprocess_plot(f, '24pt', orientation='horizontal', glyph_width=80)
    f.legend.background_fill_alpha=0.
    f.legend.border_line_alpha=0.
    #f.legend.visible=False
    f.xaxis.visible = False
    f.yaxis.visible = False


  #figs.append([fig, fig_opt])
  figs.append([fig])

bkp.show(bkl.gridplot(figs))




