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

nms = ['EGUS', 'EGS', 'ERG', 'ERL1']
figs = [[]]


#plot the KL figure
fig = bkp.figure(y_axis_type='log', plot_width=750, plot_height=750)
for i, nm in enumerate(nms):
  res = np.load('results_'+nm+'.npz')
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
  fig.line(np.arange(wt.shape[0]), rklw, color=pal[i], line_width=5, line_dash='dashed', legend=nm)
  fig.line(np.arange(wt.shape[0]), rklw_opt, color=pal[i], line_width=5, line_dash='solid', legend=nm)
figs[0].append(fig)

#plot the example set of 3-sigma ellipses
Ms = np.arange(50)
for m in Ms:
  fig = bkp.figure(x_range=(-5,5), y_range=(-5,5), plot_width=750, plot_height=750)
  #plot the data
  fig.scatter(x[:, 0], x[:, 1], fill_color='black', alpha=0.1)
  for i, nm in enumerate(nms):
    res = np.load('results_'+nm+'.npz')
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
    plot_gaussian(fig, muwt[m,:], Sigwt[m,:,:], Sig, pal[i], 5, 3, 1, 1, 'dashed', nm)
    plot_gaussian(fig, muwt_opt[m,:], Sigwt_opt[m,:], Sig, pal[i], 5, 3, 1, 1, 'solid', nm)
  figs[0].append(fig)


nm = 'EGUS'
res = np.load('results_'+ nm +'.npz')
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

for m in range(wt.shape[0]):
  fig = bkp.figure(x_range=(-5,5), y_range=(-5,5), plot_width=750, plot_height=750)
  fig_opt = bkp.figure(x_range=(-5,5), y_range=(-5,5), plot_width=750, plot_height=750)
  for (f, w, muw, Sigw) in [(fig, wt, muwt, Sigwt), (fig_opt, wt_opt, muwt_opt, Sigwt_opt)]:
    f.scatter(x[:, 0], x[:, 1], fill_color='black', alpha=0.1)
    f.scatter(x[:, 0], x[:, 1], fill_color='black', size=20*wt[m,:]/wt[m,:].max())
    plot_gaussian(f, mup, Sigp, Sig, 'black', 5, 3, 1, 1, 'solid', nm)
    plot_gaussian(f, muw[m,:], Sigw[m,:], Sig, 'green', 5, 3, 1, 1, 'solid', nm)

  figs.append([fig, fig_opt])

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

