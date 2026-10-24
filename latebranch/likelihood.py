"""Beta-binomial likelihood; logit conversion and reductions use float64."""
import numpy as np
from scipy.special import betaln, expit, gammaln

def nll_numpy(k, n, tau, rho):
    k, n = np.asarray(k, np.float64), np.asarray(n, np.float64)
    p = expit(np.clip(np.asarray(tau, np.float64), -20, 20))
    concentration = (1 - np.asarray(rho, np.float64)) / rho
    a, b = p * concentration, (1 - p) * concentration
    return -(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
             + betaln(k + a, n - k + b) - betaln(a, b))

def nll_torch(k, n, tau, rho):
    import torch
    # Casting the logit BEFORE sigmoid avoids BF16 probability quantization.
    # FP64 lgamma also prevents cancellation at high sequencing depths.
    k, n = k.double(), n.double()
    p = torch.sigmoid(tau.double().clamp(-20, 20))
    concentration = (1 - rho.double()) / rho.double()
    a, b = p * concentration, (1 - p) * concentration
    return -(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
             + torch.lgamma(k + a) + torch.lgamma(n - k + b)
             - torch.lgamma(n + a + b) - torch.lgamma(a) - torch.lgamma(b)
             + torch.lgamma(a + b))
