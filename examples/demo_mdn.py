import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from cdest.density_estimator.MDN import MixtureDensityNetwork

X, _ = make_moons(n_samples=3000, noise=.05)
X, Y = X[:,0:1], X[:,1:2]

model = MixtureDensityNetwork(ndim_x=1, ndim_y=1, x_noise_std=0.0001, y_noise_std=0.0001, random_seed=None)

model.fit(X, Y)

##################################################################################################

ymin, ymax = -1, 1.5
y_query = np.arange(ymin, ymax, 0.01)
x_cond = np.zeros((len(y_query), 1)) + 0.5
prob = model.pdf(x_cond, y_query)

##################################################################################################

fig, ax = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=[1,5], wspace=0.1))
ax[0].set_ylim([ymin, ymax])
ax[0].plot(prob,y_query)

ax[1].set_ylim([ymin, ymax])
ax[1].axes.yaxis.set_visible(False)
ax[1].scatter(X[:,0], Y)
ax[1].scatter(x_cond[:,0], y_query, s=1)

plt.show()
