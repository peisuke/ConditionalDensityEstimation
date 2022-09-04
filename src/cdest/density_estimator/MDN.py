import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, OneHotCategorical
from cdest.density_estimator.BaseDensityEstimator import BaseDensityEstimator

class BaseNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, n_components, hidden_dim=None):
        super(BaseNetwork, self).__init__()

        self.n_components = n_components
        
        if hidden_dim is None:
            hidden_dim=in_dim

        self.pi_network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n_components)
        )

        self.normal_network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * out_dim * n_components)
        )

    def forward(self, x):
        pi = self.pi_network(x)

        prm = self.normal_network(x)
        mean, std = torch.split(prm, prm.shape[1] // 2, dim=1)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
        std = torch.stack(std.split(std.shape[1] // self.n_components, 1))

        mean = mean.transpose(0, 1)
        std = (F.elu(std)+1+1e-7).transpose(0, 1)
        
        return pi, mean, std


class MixtureDensityNetwork(BaseDensityEstimator):
    def __init__(self, ndim_x=None, ndim_y=None, n_gaussians=10, model=None, n_epoch=1000, 
                 x_noise_std=None, y_noise_std=None, random_seed=None, verbose=-1):
        self.ndim_x = ndim_x
        self.ndim_y = ndim_y
        self.model = model
        self.n_gaussians = n_gaussians
        self.n_epoch = n_epoch
        self.x_noise_std = x_noise_std
        self.y_noise_std = y_noise_std
        self.verbose = verbose

        self._build_model()

    def _build_model(self):
        if self.model is None:
            self.model = BaseNetwork(self.ndim_x, self.ndim_y, self.n_gaussians)

    def fit(self, X, Y, **kwargs):
        X = self._handle_input_dimensionality(X)

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
            pi_logits, mean, std = self.model(X)

            normal = Normal(mean, std)
            loglik = normal.log_prob(Y.unsqueeze(1).expand_as(normal.loc))
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
        pi_logits, mean, std = self.model(torch.tensor(X.astype(np.float32)))
        pi = OneHotCategorical(logits=pi_logits)
        normal = Normal(mean, std)

        loglik = normal.log_prob(torch.tensor(Y.astype(np.float32)).unsqueeze(1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=2)
        log_pdf = torch.logsumexp(pi.logits + loglik, dim=1)

        return log_pdf.numpy()
