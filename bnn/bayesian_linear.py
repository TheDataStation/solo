import math
import torch
import torch.nn as nn

class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size())
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input_x):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input_x - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class GaussianPrior(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.gaussian = torch.distributions.Normal(mu,sigma)
    
    def log_prob(self, input_x):
        return self.gaussian.log_prob(input_x)


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, 
                 weight_mu_prior, weight_sigma_prior,
                 bias_mu_prior, bias_sigma_prior):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = GaussianPrior(weight_mu_prior, weight_sigma_prior)
        self.bias_prior = GaussianPrior(bias_mu_prior, bias_sigma_prior) 
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input_x, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input_x, weight, bias)


