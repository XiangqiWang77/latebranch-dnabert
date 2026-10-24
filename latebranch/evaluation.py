"""Training-only dispersions, calibration-only scales, and descriptive metrics."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr

from .likelihood import nll_numpy


def fit_rho(training, assays):
    result = []
    for assay in assays:
        rows = training[training.assay == assay]
        if len(rows) < 30:
            raise ValueError(f"At least 30 training observations required to estimate rho for {assay}")
        k = rows.n_alt.to_numpy(); n = (rows.n_alt + rows.n_ref).to_numpy()
        fitted = minimize_scalar(lambda x: nll_numpy(k, n, 0., np.exp(x)).mean(),
                                 bounds=(np.log(1e-4), np.log(.8)), method="bounded", options={"xatol":1e-8})
        if not fitted.success:
            raise RuntimeError(f"Training dispersion optimization failed for {assay}")
        result.append(float(np.exp(fitted.x)))
    return result


def fit_calibration(predictions, config):
    required = {"split", "assay", "n_ref", "n_alt", "tau_raw", "rho"}
    if not required.issubset(predictions.columns) or not predictions.split.eq("calibration").all():
        raise ValueError("Use only predictions explicitly labeled split=calibration")
    if set(predictions.assay) != set(config.assays):
        raise ValueError("Calibration assays must match the checkpoint vocabulary")
    scales = []
    for assay, rho in zip(config.assays, config.rho):
        rows = predictions[predictions.assay == assay]
        if len(rows) < 30 or not np.allclose(rows.rho, rho, rtol=0, atol=1e-14):
            raise ValueError(f"Insufficient support or mismatched rho for {assay}")
        k = rows.n_alt.to_numpy(); n = (rows.n_alt+rows.n_ref).to_numpy(); tau = rows.tau_raw.to_numpy()
        if not np.isfinite(tau).all() or not np.isfinite(k).all() or not np.isfinite(n).all() or np.any((k < 0)|(k > n)|(n <= 0)|(k != np.floor(k))|(n != np.floor(n))):
            raise ValueError("Invalid calibration predictions/counts")
        objective = lambda scale: float(nll_numpy(k, n, scale*tau, rho).mean())
        fitted = minimize_scalar(objective, bounds=(0,3), method="bounded")
        if not fitted.success:
            raise RuntimeError(f"Scale optimization failed for {assay}")
        candidates = [(fitted.fun,float(fitted.x)), (objective(0),0.), (objective(3),3.)]
        scales.append(min(candidates)[1])
    return dict(assays=config.assays, rho=config.rho, scales=scales,
                scope="Primary fixed-rho scale calibration; calibration observations only")


def evaluate(predictions):
    required = {"assay", "n_ref", "n_alt", "rho", "tau_raw", "tau"}
    if predictions.empty or not required.issubset(predictions.columns):
        raise ValueError("Prediction CSV must contain counts, rho and raw/calibrated logits")
    output = []
    groups = [("all", predictions)] + list(predictions.groupby("assay", sort=True))
    for label, rows in groups:
        k = rows.n_alt.to_numpy(float); n = (rows.n_alt+rows.n_ref).to_numpy(float)
        rho = rows.rho.to_numpy(float)
        if np.any((n <= 0)|(k < 0)|(k > n)|(rho <= 0)|(rho >= 1)|(k != np.floor(k))|(n != np.floor(n))) or not np.isfinite(np.concatenate([k,n,rho])).all():
            raise ValueError("Invalid counts or dispersion")
        observed = np.log((k+.5)/(n-k+.5))
        for mode, key in (("raw","tau_raw"),("calibrated","tau")):
            tau = rows[key].to_numpy(float)
            if not np.isfinite(tau).all():
                raise ValueError("Nonfinite predicted log ratio")
            use = 2*k != n
            signs = np.sign(tau[use])*np.sign(2*k[use]-n[use])
            varied = len(tau)>1 and np.ptp(tau)>0 and np.ptp(observed)>0
            output.append(dict(assay=label, mode=mode, rows=len(rows),
                nll=float(nll_numpy(k,n,tau,rho).mean()), zero_effect_nll=float(nll_numpy(k,n,0.,rho).mean()),
                pearson=float(pearsonr(tau,observed).statistic) if varied else np.nan,
                spearman=float(spearmanr(tau,observed).statistic) if varied else np.nan,
                direction_accuracy=float(((signs>0)+.5*(signs==0)).mean()) if use.any() else np.nan,
                direction_rows=int(use.sum())))
    return pd.DataFrame(output)
