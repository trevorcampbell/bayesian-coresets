import bokeh.layouts as bkl
from utils import *
import bokeh.plotting as bkp
import numpy as np

size_x_axis = True
trial_num = 3
nm = ('EGUS', 'Uniform')
Ms = [0, 1, 2, 5, 10, 20, 50, 99]

nm = ('ERCG', 'CorrectiveGreedy')
Ms = np.arange(20)

#nm = ('ERG', 'Greedy')
#Ms = np.arange(20)

#nm = ('ERL1', 'L1')
#Ms = np.arange(20)

#nm = ('ERL1U', 'L1U')
#Ms = np.arange(20)






#plot the KL figure

#plot the sequence of coreset pts and comparison of nonopt + opt
res = np.load('results_'+nm[0] + '_' + str(trial_num)+'.npz')
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

figs = []
for m in Ms:
  fig = bkp.figure(x_range=(-5,5), y_range=(-5,5), plot_width=750, plot_height=750)
  fig_opt = bkp.figure(x_range=(-5,5), y_range=(-5,5), plot_width=750, plot_height=750)
  for f in [fig, fig_opt]:
    preprocess_plot(f, '24pt', False)

  for (f, w, muw, Sigw) in [(fig, wt, muwt, Sigwt), (fig_opt, wt_opt, muwt_opt, Sigwt_opt)]:
    msz = np.where((w > 0).sum(axis=1) <= m)[0][-1]
    f.scatter(x[:, 0], x[:, 1], fill_color='black', alpha=0.1)

    if size_x_axis:
      f.scatter(x[:, 0], x[:, 1], fill_color='black', size=20*w[msz,:]/w[msz,:].max())
    else:
      f.scatter(x[:, 0], x[:, 1], fill_color='black', size=20*w[m,:]/w[m,:].max())

    plot_gaussian(f, mup, Sigp, Sig, 'black', 5, 3, 1, 1, 'solid', 'Exact')

    if size_x_axis:
      plot_gaussian(f, muw[msz,:], Sigw[msz,:], Sig, 'green', 5, 3, 1, 1, 'solid', nm[1]+', size ' + str( (w[msz, :]>0).sum() ))
    else:
      plot_gaussian(f, muw[m,:], Sigw[m,:], Sig, 'green', 5, 3, 1, 1, 'solid', nm[1]+', ' + str(m) +' itrs') 


  for f in [fig, fig_opt]:
    postprocess_plot(f, '16pt', orientation='horizontal', glyph_width=40)

  figs.append([fig, fig_opt])

bkp.show(bkl.gridplot(figs))




