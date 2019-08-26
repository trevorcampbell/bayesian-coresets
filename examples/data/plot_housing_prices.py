import numpy as np
import bokeh.plotting as bkp


data = np.load('prices2018.npy')

data = data[np.argsort(data[:, 2]), :]

p = 3
c = ((np.log(data[:,2]) - np.log(data[:,2]).min()) / (np.log(data[:,2]).max() - np.log(data[:,2]).min()))**p

#c = np.linspace(0, 1, data.shape[0])

colors = ['#%02x%02x%02x' % (int(r), int(b), int(g)) for (r, b, g) in zip(255*c, 0*np.ones(c.shape[0]), 255*(1.-c))]

p = bkp.figure()
p.scatter(data[:, 1], data[:, 0], fill_color=colors, line_color=None, fill_alpha=0.01)
bkp.show(p)
