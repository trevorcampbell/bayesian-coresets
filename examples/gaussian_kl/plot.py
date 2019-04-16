import bokeh.plotting as bkp
import bokeh.io as bki
import bokeh.layouts as bkl
from bokeh.models import FuncTickFormatter
import bokeh.palettes 
import numpy as np

def plot_gaussian(plot, mup, Sigp, Sig, color, dotsize, linewidth, dotalpha, linealpha, line_dash, name):
  plot.circle(mup[0], mup[1], color=color, size=dotsize, alpha=dotalpha, legend=name)
  t = np.linspace(0., 2*np.pi, 100)
  t = np.array([np.cos(t), np.sin(t)])
  t = 3*np.linalg.cholesky(Sigp+Sig).dot(t) + mup[:, np.newaxis]
  plot.line(t[0, :], t[1, :], color=color, line_width=linewidth, alpha=linealpha, line_dash=line_dash, legend=name)

def plot_meanstd(plot, x, ys, color, linewidth, alpha, line_dash, name):
  plot.line(x, ys.mean(axis=0), color=color, line_width=linewidth, line_dash=line_dash, legend=nm)
  plot.patch(np.hstack((x, x[::-1])), np.hstack(( ys.mean(axis=0)-ys.std(axis=0), (ys.mean(axis=0)+ys.std(axis=0))[::-1] )), color=color, line_width=linewidth/2, line_dash=line_dash, alpha=alpha, legend=nm)

def plot_medianquartiles(plot, x, ys, color, linewidth, alpha, line_dash, name):
  ys25 = np.percentile(ys, 49, axis=0)
  ys50 = np.percentile(ys, 50, axis=0)
  ys75 = np.percentile(ys, 51, axis=0)
  plot.line(x, ys25, color=color, line_width=linewidth, line_dash=line_dash, legend=nm)
  plot.line(x, ys50, color=color, line_width=linewidth, line_dash=line_dash, legend=nm)
  plot.line(x, ys75, color=color, line_width=linewidth, line_dash=line_dash, legend=nm)
  #plot.patch(np.hstack((x, x[::-1])), np.hstack(( ys25, ys75[::-1] )), color=color, line_width=linewidth/2, line_dash=line_dash, alpha=alpha, legend=nm)



#logFmtr = FuncTickFormatter(code="return Math.log10(tick)")
logFmtr = FuncTickFormatter(code="""
var trns = [
'\u2070',
'\u00B9',
'\u00B2',
'\u00B3',
'\u2074',
'\u2075',
'\u2076',
'\u2077',
'\u2078',
'\u2079']
if (Math.log10(tick) < 0){
  return '10\u207B'+trns[Math.round(Math.abs(Math.log10(tick)))];
} else {
  return '10'+trns[Math.round(Math.abs(Math.log10(tick)))];
}
""")

pal = bokeh.palettes.colorblind['Colorblind'][8]
pl = [pal[0], pal[1], pal[3]]
pl.extend(pal[4:8])
pal = pl

null_font_size='0pt'
axis_font_size='40pt'

plot_reverse_kl = False
size_x_axis = False
scaled = True
n_trials = 100
nms = [('EGUS', 'Uniform'), ('ERG', 'Greedy'), ('ERL1', 'L1')]
Ms_2d = [0, 1, 2, 5, 10, 20, 50, 99]
Ms_kl = np.arange(100)

#plot the KL figure
figs = []
fig = bkp.figure(y_axis_type='log', plot_width=750, plot_height=750)
#fig = bkp.figure(plot_width=750, plot_height=750)
for i, nm in enumerate(nms):
  kl = []
  klopt = []
  for t in range(n_trials):
    res = np.load('results_'+nm[0]+'_'+('scaled' if scaled else 'unscaled') + '_' + str(t)+'.npz')
    if plot_reverse_kl:
      klt = res['rklw']
      kloptt = res['rklw_opt']
    else:
      klt = res['fklw']
      kloptt = res['fklw_opt']
    sz = (res['w'] > 0).sum(axis=1) 
    szopt = (res['w_opt'] > 0).sum(axis=1) 
    if size_x_axis:
      kl.append(np.interp(Ms_kl, sz, klt))
      klopt.append(np.interp(Ms_kl, szopt, kloptt))
    else:
      kl.append(klt)
      klopt.append(kloptt)
    #fig.scatter(sz[-1], kl[-1], color=pal[i], legend=nm) 
    #fig.scatter(szopt[-1], klopt[-1], color=pal[i], legend=nm) 
  if size_x_axis:
    fig.line(Ms_kl, np.percentile(kl, 50, axis=0), color=pal[i], line_width=5, line_dash='dashed', legend=nm[1]) 
    fig.line(Ms_kl, np.percentile(klopt, 50, axis=0), color=pal[i], line_width=5, line_dash='solid', legend=nm[1]) 
  else:
    kl = np.array(kl)
    klopt = np.array(klopt)
    fig.line(np.arange(kl.shape[1]), np.percentile(kl, 50, axis=0), color=pal[i], line_width=5, line_dash='dashed', legend=nm[1])
    fig.line(np.arange(kl.shape[1]), np.percentile(klopt, 50, axis=0), color=pal[i], line_width=5, line_dash='solid', legend=nm[1])
    #plot_meanstd(fig, np.arange(kl.shape[1]), klopt, pal[i], 5, 0.3, 'solid', nm)
figs.append([fig])

#plot the example set of 3-sigma ellipses
figs.append([])
trial_num = 0
for m in Ms_2d:
  fig = bkp.figure(x_range=(-5,5), y_range=(-5,5), plot_width=750, plot_height=750)
  #plot the data
  x = np.load('results_EGUS_'+('scaled' if scaled else 'unscaled') + '_' + str(trial_num)+'.npz')['x']
  fig.scatter(x[:, 0], x[:, 1], fill_color='black', alpha=0.1)
  for i, nm in enumerate(nms):
    res = np.load('results_'+nm[0]+'_'+('scaled' if scaled else 'unscaled') + '_' + str(trial_num)+'.npz')
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
    rklw = res['rklw']
    fklw = res['fklw']
    rklw_opt = res['rklw_opt']
    fklw_opt = res['fklw_opt']
    plot_gaussian(fig, mup, Sigp, Sig, 'black', 5, 3, 1, 1, 'solid', 'True')
    plot_gaussian(fig, muwt[m,:], Sigwt[m,:,:], Sig, pal[i], 5, 3, 1, 1, 'dashed', nm[1])
    plot_gaussian(fig, muwt_opt[m,:], Sigwt_opt[m,:], Sig, pal[i], 5, 3, 1, 1, 'solid', nm[1])
  figs[1].append(fig)
#
#
##plot the sequence of coreset pts and comparison of nonopt + opt
#nm = 'EGUS'
#trial_num = 0
#res = np.load('results_'+nm+'_'+('scaled' if scaled else 'unscaled') + '_' + str(trial_num)+'.npz')
#x = res['x']
#wt = res['w']
#wt_opt = res['w_opt']
#Sig = res['Sig']
#mup = res['mup']
#Sigp = res['Sigp']
#muwt = res['muw']
#Sigwt = res['Sigw']
#muwt_opt = res['muw_opt']
#Sigwt_opt = res['Sigw_opt']
#rklw = res['rklw']
#fklw = res['fklw']
#rklw_opt = res['rklw_opt']
#fklw_opt = res['fklw_opt']
#
#for m in Ms:
#  fig = bkp.figure(x_range=(-5,5), y_range=(-5,5), plot_width=750, plot_height=750)
#  fig_opt = bkp.figure(x_range=(-5,5), y_range=(-5,5), plot_width=750, plot_height=750)
#  for (f, w, muw, Sigw) in [(fig, wt, muwt, Sigwt), (fig_opt, wt_opt, muwt_opt, Sigwt_opt)]:
#    f.scatter(x[:, 0], x[:, 1], fill_color='black', alpha=0.1)
#    f.scatter(x[:, 0], x[:, 1], fill_color='black', size=20*w[m,:]/w[m,:].max())
#    plot_gaussian(f, mup, Sigp, Sig, 'black', 5, 3, 1, 1, 'solid', nm)
#    plot_gaussian(f, muw[m,:], Sigw[m,:], Sig, 'green', 5, 3, 1, 1, 'solid', nm)
#
#  figs.append([fig, fig_opt])
#
#
###TODO plot the sequence of coreset pts and comparison of scaled vs unscaled
##nm = 'ERL1'
##trial_num = 0
##res_scaled = np.load('results_'+nm+'_scaled_' + str(trial_num)+'.npz')
##res_unscaled = np.load('results_'+nm+'_unscaled_' + str(trial_num)+'.npz')
##x = res_unscaled['x']
##Sig = res_unscaled['Sig']
##mup = res_unscaled['mup']
##Sigp = res_unscaled['Sigp']
##
##wt_opt_unscaled = res_unscaled['w_opt']
##muwt_opt_unscaled = res_unscaled['muw_opt']
##Sigwt_opt_unscaled = res_unscaled['Sigw_opt']
##
##wt_opt_scaled = res_scaled['w_opt']
##muwt_opt_scaled = res_scaled['muw_opt']
##Sigwt_opt_scaled = res_scaled['Sigw_opt']
##
##for m in Ms:
##  fig_unscaled = bkp.figure(x_range=(-5,5), y_range=(-5,5), plot_width=750, plot_height=750)
##  fig_scaled = bkp.figure(x_range=(-5,5), y_range=(-5,5), plot_width=750, plot_height=750)
##  for (f, w, muw, Sigw) in [(fig_unscaled, wt_opt_unscaled, muwt_opt_unscaled, Sigwt_opt_unscaled), (fig_scaled, wt_opt_scaled, muwt_opt_scaled, Sigwt_opt_scaled)]:
##    f.scatter(x[:, 0], x[:, 1], fill_color='black', alpha=0.1)
##    f.scatter(x[:, 0], x[:, 1], fill_color='black', size=20*w[m,:]/w[m,:].max())
##    plot_gaussian(f, mup, Sigp, Sig, 'black', 5, 3, 1, 1, 'solid', nm)
##    plot_gaussian(f, muw[m,:], Sigw[m,:], Sig, 'green', 5, 3, 1, 1, 'solid', nm)
##
##  figs.append([fig, fig_opt])

bkp.show(bkl.gridplot(figs))



##old plotting code
##import bokeh.plotting as bkp
##import bokeh.io as bki
##import bokeh.layouts as bkl
##from bokeh.models import FuncTickFormatter
##import bokeh.palettes 
##import numpy as np
##
##def plot_gaussian(plot, mup, Sigp, Sig, color, dotsize, linewidth, dotalpha, linealpha, name):
##  plot.circle(mup[0], mup[1], color=color, size=dotsize, alpha=dotalpha, legend=name)
##  t = np.linspace(0., 2*np.pi, 100)
##  t = np.array([np.cos(t), np.sin(t)])
##  t = 3*np.linalg.cholesky(Sigp+Sig).dot(t) + mup[:, np.newaxis]
##  plot.line(t[0, :], t[1, :], color=color, line_width=linewidth, alpha=linealpha, legend=name)
##
###logFmtr = FuncTickFormatter(code="return Math.log10(tick)")
##logFmtr = FuncTickFormatter(code="""
##var trns = [
##'\u2070',
##'\u00B9',
##'\u00B2',
##'\u00B3',
##'\u2074',
##'\u2075',
##'\u2076',
##'\u2077',
##'\u2078',
##'\u2079']
##if (Math.log10(tick) < 0){
##  return '10\u207B'+trns[Math.round(Math.abs(Math.log10(tick)))];
##} else {
##  return '10'+trns[Math.round(Math.abs(Math.log10(tick)))];
##}
##""")
##
##pal = bokeh.palettes.colorblind['Colorblind'][8]
##
##null_font_size='0pt'
##axis_font_size='40pt'
##
##figs = []
##
###command that generated the results files
###np.savez('results.npz', x=x, th0=th0, Sig0=Sig0, Sig=Sig, mup=mup, Sigp=Sigp, lambdas=lambdas, 
###                        w_l1=w_l1, w_l1_post=w_l1_post, w_g=w_g, w_g_post=w_g_post, 
###                        muw_l1=muw_l1, muw_l1_post=muw_l1_post, muw_g=muw_g, muw_g_post = muw_g_post,
###                        Sigw_l1=Sigw_l1, Sigw_l1_post = Sigw_l1_post, Sigw_g=Sigw_g, Sigw_g_post=Sigw_g_post,
###                        kl_g=kl_g, kl_g_post=kl_g_post, kl_l1=kl_l1, kl_l1_post=kl_l1_post)
##
##res = np.load('results_EGS.npz')
##x = res['x']
##wt = res['w']
##wt_opt = res['w_opt']
##Sig = res['Sig']
##mup = res['mup']
##Sigp = res['Sigp']
##muwt = res['muw']
##Sigwt = res['Sigw']
##muwt_opt = res['muw_opt']
##Sigwt_opt = res['Sigw_opt']
##rklw = res['rklw']
##fklw = res['fklw']
##rklw_opt = res['rklw_opt']
##fklw_opt = res['fklw_opt']
##
##for (kl, klopt, klnm) in [(rklw, rklw_opt, 'Reverse KL'), (fklw, fklw_opt, 'Forward KL')]:
##  fig = bkp.figure(plot_width=750, plot_height=750)
##  fig.line(np.arange(wt.shape[0]), kl, color='green', line_width=5, legend=klnm)
##  fig.line(np.arange(wt.shape[0]), klopt, color='blue', line_width=5, legend=klnm+' Optimized')
##  figs.append([fig])
##
##for m in range(wt.shape[0]):
##  fig = bkp.figure(x_range=(-5,5), y_range=(-5,5), plot_width=750, plot_height=750)
##  fig_opt = bkp.figure(x_range=(-5,5), y_range=(-5,5), plot_width=750, plot_height=750)
##  for (f, w, muw, Sigw) in [(fig, wt, muwt, Sigwt), (fig_opt, wt_opt, muwt_opt, Sigwt_opt)]:
##    f.scatter(x[:, 0], x[:, 1], fill_color='black', alpha=0.1)
##    f.scatter(x[:, 0], x[:, 1], fill_color='black', size=20*w[m,:]/w[m,:].max())
##    plot_gaussian(f, mup, Sigp, Sig, 'blue', 5, 1, 1, 1, 'True')
##    plot_gaussian(f, muw[m,:], Sigw[m,:], Sig, 'green', 5, 1, 1, 1, 'Coreset')
##
##  figs.append([fig, fig_opt])
##
##bkp.show(bkl.gridplot(figs))

