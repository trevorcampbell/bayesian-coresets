import numpy as np
import bokeh.plotting as bkp
import bokeh.io as bki
import bokeh.layouts as bkl
from bokeh.models import FuncTickFormatter
import bokeh.models as bkm
import bokeh.palettes 
import time
import os

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
'\u2079'];
var tick_power = Math.floor(Math.log10(tick));
var tick_mult = Math.pow(10, Math.log10(tick) - tick_power);
var ret = '';
if (tick_mult > 1.) {
  if (Math.abs(tick_mult - Math.round(tick_mult)) > 0.05){
    ret = tick_mult.toFixed(1) + '\u22C5';
  } else {
    ret = tick_mult.toFixed(0) +'\u22C5';
  }
}
ret += '10';
if (tick_power < 0){
  ret += '\u207B';
  tick_power = -tick_power;
}
power_digits = []
while (tick_power > 9){
  power_digits.push( tick_power - Math.floor(tick_power/10)*10 )
  tick_power = Math.floor(tick_power/10)
}
power_digits.push(tick_power)
for (i = power_digits.length-1; i >= 0; i--){
  ret += trns[power_digits[i]];
}
return ret;
""")

#pal = [pal[0], pal[1], '#d62728', pal[3], pal[4], pal[5], pal[6], pal[7], pal[2]]

pal = bokeh.palettes.colorblind['Colorblind'][8]
pl = [pal[0], pal[1], pal[3]]
pl.extend(pal[4:8])
pl.append('#d62728')
pal = pl


dnames = ['lr_synth', 'lr_ds1', 'lr_phishing', 'poiss_synth', 'poiss_biketrips', 'poiss_airportdelays']
dnames = ['lr_synth', 'poiss_synth']
algs = [('uniform', 'Uniform', pal[7]), ('hilbert','GIGA (noisy)', pal[5]), ('hilbert_corr', 'Fully Corrective GIGA (noisy)', pal[1]), ('riemann', 'Greedy', pal[3]), ('riemann_corr', 'Fully Corrective Greedy', pal[4]),('hilbert_good','GIGA (truth)', pal[2]), ('hilbert_corr_good', 'Fully Corrective GIGA (truth)', pal[0])]

fig = bkp.figure(y_axis_type='log', y_axis_label='Reverse KL', x_axis_type='log', x_axis_label='Coreset Size', width=2000, height=2000)
fig2 = bkp.figure(y_axis_type='log', y_axis_label='Reverse KL', x_axis_type='log', x_axis_label='CPU Time (s)', width=2000, height=2000)

for f in [fig, fig2]:
  axis_font_size='12pt'
  legend_font_size='12pt'
  f.xaxis.axis_label_text_font_size= axis_font_size
  f.xaxis.major_label_text_font_size= axis_font_size
  f.yaxis.axis_label_text_font_size= axis_font_size
  f.yaxis.major_label_text_font_size= axis_font_size
  f.yaxis.formatter = logFmtr
  f.xaxis.formatter = logFmtr

dnmsalgs = [(dnm, alg) for dnm in dnames for alg in algs]

#get uniform median normalization
std_kls = {}
for didx, dnm in enumerate(dnames):
  trials = [fn for fn in os.listdir('.') if dnm+'_uniform_results_' in fn]
  if len(trials) == 0: 
    print('Need to run uniform to establish baseline first')
    quit()
  kltot = 0.
  for tridx, fn in enumerate(trials):
    res = np.load(fn)
    kltot += np.log(res['kls']).mean()
  std_kls[dnm] = np.exp(kltot / len(trials))

for idx, zppd in enumerate(dnmsalgs):
  dnm, alg = zppd
  trials = [fn for fn in os.listdir('.') if dnm+'_'+alg[0]+'_results_' in fn]
  if len(trials) == 0: continue
  Ms = np.load(trials[0])['Ms']
  kls = np.zeros((len(trials), len(Ms)))
  cputs = np.zeros((len(trials), len(Ms)))
  kl0 = std_kls[dnm] 
  for tridx, fn in enumerate(trials):
    #np.savez(fldr+'_'+dnm+'_'+alg+'_results_'+str(ID)+'.npz', cputs=cputs, wts=wts, Ms=Ms, mus=mus_laplace, Sigs=Sigs_laplace, kls=kls_laplace)
    res = np.load(fn)
    cput = res['cputs']
    #if cput.shape[0] != len(Ms):
    #  print(fn)
    cputs[tridx, :] = cput[:len(Ms)]
    wts = res['wts']
    mu = res['mus']
    Sig = res['Sigs']
    kl = res['kls']
    kls[tridx, :] = kls[:len(Ms)]/kl0
   

  cput50 = np.percentile(cputs, 50, axis=0)
  cput25 = np.percentile(cputs, 25, axis=0)
  cput75 = np.percentile(cputs, 75, axis=0)

  kl50 = np.percentile(kls, 50, axis=0)
  kl25 = np.percentile(kls, 25, axis=0)
  kl75 = np.percentile(kls, 75, axis=0)


  #fig.line(Ms, kls.mean(axis=0), color=alg[2], legend=alg[1], line_width=10)
  fig.line(Ms, kl50, color=alg[2], legend=alg[1], line_width=10)
  #fig.line(Ms, kls.mean(axis=0)+kls.std(axis=0), color=alg[2], legend=alg[1], line_dash='dashed')
  #fig.line(Ms, kls.mean(axis=0)-kls.std(axis=0), color=alg[2], legend=alg[1], line_dash='dashed')

  #fig2.circle(cputs.mean(axis=0), kls.mean(axis=0), color=alg[2], legend=alg[1], size=25)
  #fig2.segment(x0=cputs.mean(axis=0)-cputs.std(axis=0), x1 = cputs.mean(axis=0)+cputs.std(axis=0), y0 = kls.mean(axis=0), y1 = kls.mean(axis=0), color=alg[2], legend=alg[1], line_width=4)
  #fig2.segment(y0=kls.mean(axis=0)-kls.std(axis=0), y1 = kls.mean(axis=0)+kls.std(axis=0), x0 = cputs.mean(axis=0), x1 = cputs.mean(axis=0), color=alg[2], legend=alg[1], line_width=4)

  fig2.circle(cput50, kl50, color=alg[2], legend=alg[1], size=25)
  fig2.segment(x0=cput25, x1 = cput75, y0 = kl50, y1 = kl50, color=alg[2], legend=alg[1], line_width=4)
  fig2.segment(y0=kl25, y1 = kl75, x0 = cput50, x1 = cput50, color=alg[2], legend=alg[1], line_width=4)

   
#rndlbl = bkm.Label(x=1.0, x_offset=-10, y=700, y_units='screen', text='Full Dataset MCMC', angle=90, angle_units='deg', text_font_size='30pt')
#rndspan = bkm.Span(location = 1.0, dimension='height', line_width=8, line_color='black', line_dash='40 40')
#fig_cput.add_layout(rndspan)
#fig_cput.add_layout(rndlbl)

for f in [fig, fig2]:
  f.legend.label_text_font_size= legend_font_size
  f.legend.glyph_width=40
  f.legend.glyph_height=80
  f.legend.spacing=20
  f.legend.orientation='horizontal'

bkp.show(bkl.gridplot([[fig, fig2]]) )

