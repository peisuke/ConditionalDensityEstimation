# ConditionalDensityEstimation

```
from sklearn.datasets import make_moons
from cde.density_estimator.LSCDE import LSConditionalDensityEstimation
from cde.density_estimator.MDN import MixtureDensityNetwork
from cde.density_estimator.KMN import KernelMixtureNetwork
import numpy as np

X, _ = make_moons(n_samples=3000, noise=.05)
X, Y = X[:,0:1], X[:,1:2]

model = KernelMixtureNetwork(ndim_x=1, ndim_y=1,  n_centers=100, bandwidth=0.05, random_seed=None)
#model = KernelMixtureNetwork(ndim_x=1, ndim_y=1, x_noise_std=0.0001, y_noise_std=0.0001, random_seed=None)
#model = LSConditionalDensityEstimation(ndim_x=1, ndim_y=1,  bandwidth=0.05, random_seed=None)

model.fit(X, Y)

# predict pdf
y_query = np.arange(-1, 1.5, 0.01)
x_cond = np.zeros((len(y_query), 1)) + 0.5
prob = model.pdf(x_cond, y_query)
```
