import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal, OneHotCategorical
from cdest.density_estimator.BaseDensityEstimator import BaseDensityEstimator
from cdest.utils.center_point_select import sample_center_points

class BaseNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, n_centers, hidden_dim=None):
        super(BaseNetwork, self).__init__()

        self.n_centers = n_centers
        
        if hidden_dim is None:
            hidden_dim=10

        ## centerの数だけの出力
        self.pi_network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n_centers)
        )

    def forward(self, x):
        pi = self.pi_network(x)
        return pi


class KernelMixtureNetwork(BaseDensityEstimator):
    def __init__(self, ndim_x=None, ndim_y=None, center_sampling_method='k_means', 
                 bandwidth=0.05, n_centers=20, keep_edges=True, model=None, n_epoch=1000, 
                 x_noise_std=None, y_noise_std=None, random_seed=None, verbose=-1):
        self.ndim_x = ndim_x
        self.ndim_y = ndim_y
        self.center_sampling_method = center_sampling_method
        self.bandwidth = bandwidth
        self.n_centers = n_centers
        self.keep_edges = keep_edges
        self.model = model
        self.n_epoch = n_epoch
        self.x_noise_std = x_noise_std
        self.y_noise_std = y_noise_std
        self.random_state = np.random.RandomState(seed=random_seed)
        self.verbose = verbose
        self.sampled_locs = None
        self._build_model()

    def _build_model(self):
        if self.model is None:
            self.model = BaseNetwork(self.ndim_x, self.ndim_y, self.n_centers)

    def fit(self, X, Y, **kwargs):
        X = self._handle_input_dimensionality(X)

        sampled_locs = sample_center_points(Y, method=self.center_sampling_method, k=self.n_centers,
                                            keep_edges=self.keep_edges, random_state=self.random_state)
        self.sampled_locs = torch.tensor(sampled_locs.astype(np.float32))

        X = torch.tensor(X.astype(np.float32))
        Y = torch.tensor(Y.astype(np.float32))
      
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        for i in range(self.n_epoch):
            optimizer.zero_grad()
           
            if self.x_noise_std is not None:
                noise_x = torch.normal(0, self.x_noise_std, size=X.size())
                X += noise_x
            if self.y_noise_std is not None:
                noise_y = torch.normal(0, self.x_noise_std, size=Y.size())
                Y += noise_y

            pi_logits = self.model(X)

            normal = Normal(self.sampled_locs, torch.tensor(self.bandwidth))
            loglik = normal.log_prob(Y.unsqueeze(1).expand(Y.shape[:1] + normal.loc.shape))
            loglik = torch.sum(loglik, dim=2)

            pi = OneHotCategorical(logits=pi_logits)
            loss = -torch.logsumexp(pi.logits + loglik, dim=1).sum()

            loss.backward()
            optimizer.step()

            if self.verbose >= 0 and i % 100 == 0:
                print(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}")

    def pdf(self, X, Y):
        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        log_pdf = self._log_pdf(X, Y)
        return np.exp(log_pdf)

    def log_pdf(self, X, Y):
        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        return self._log_pdf(X, Y)

    @torch.no_grad()
    def _log_pdf(self, X, Y):
        X = torch.tensor(X.astype(np.float32))
        Y = torch.tensor(Y.astype(np.float32))

        pi_logits = self.model(X)
        pi = OneHotCategorical(logits=pi_logits)
        normal = Normal(self.sampled_locs, torch.tensor(self.bandwidth))

        loglik = normal.log_prob(Y.unsqueeze(1).expand(Y.shape[:1] + normal.loc.shape))
        loglik = torch.sum(loglik, dim=2)
        log_pdf = torch.logsumexp(pi.logits + loglik, dim=1)

        return log_pdf.numpy()
