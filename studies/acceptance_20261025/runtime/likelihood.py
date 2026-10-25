"""Count likelihood outside autocast, with independent SciPy verification."""
import numpy as np
from scipy.optimize import minimize_scalar, minimize
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


def fit_null_rho(frame, rows):
    """One training-only null MLE per assay, shared by all architectures."""
    result = []
    for assay in range(5):
        x = frame.iloc[rows]
        x = x[x.assay_id == assay]
        if len(x) < 30:
            raise ValueError(f"Insufficient training support for assay {assay}")
        opt = minimize_scalar(lambda log_rho: float(nll_numpy(
            x.n_alt.to_numpy(), x.total.to_numpy(), 0., np.exp(log_rho)).mean()),
            bounds=(np.log(1e-4), np.log(.8)), method="bounded",
            options={"xatol": 1e-8})
        if not opt.success:
            raise RuntimeError(opt.message)
        result.append(float(np.exp(opt.x)))
    return result


def calibrate(frame, rows, tau, rho):
    """Primary fixed-rho scaling; secondary joint scale/rho, calibration only."""
    scales, joint_scales, joint_rho = [], [], []
    diagnostics = []
    x = frame.iloc[rows]
    for assay in range(5):
        mask = x.assay_id.to_numpy() == assay
        k, n, t = x.n_alt.to_numpy()[mask], x.total.to_numpy()[mask], tau[mask]
        if len(k) < 30:
            raise ValueError(f"Insufficient calibration support for assay {assay}")
        objective = lambda a: float(nll_numpy(k, n, a * t, rho[assay]).mean())
        opt = minimize_scalar(objective, bounds=(0, 3), method="bounded")
        candidates = [(float(opt.x), float(opt.fun)), (0., objective(0.)), (3., objective(3.))]
        scale = min(candidates, key=lambda item: item[1])[0]
        joint = minimize(lambda v: float(nll_numpy(k, n, v[0] * t, np.exp(v[1])).mean()),
                         x0=[scale, np.log(rho[assay])], method="L-BFGS-B",
                         bounds=[(0, 3), (np.log(1e-4), np.log(.8))])
        if not opt.success or not joint.success or not np.isfinite(joint.fun):
            raise RuntimeError(f"Calibration failed for assay {assay}: {opt.message}; {joint.message}")
        scales.append(scale)
        joint_scales.append(float(joint.x[0]))
        joint_rho.append(float(np.exp(joint.x[1])))
        diagnostics.append({"assay": assay, "rows": int(mask.sum()), "scale_at_boundary": scale < .001 or scale > 2.999})
    return dict(scales=scales, joint_scales=joint_scales, joint_rho=joint_rho, diagnostics=diagnostics)
