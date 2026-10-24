"""Forward/reverse-complement averaging and explicit assay calibration."""
import numpy as np
import torch
from scipy.special import expit
from torch.utils.data import DataLoader

from .data import AlleleDataset, AlleleCollator


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def predict_frame(model, tokenizer, config, frame, batch_size=32, calibration=None):
    if frame.empty or batch_size < 1:
        raise ValueError("Nonempty prediction input and positive batch size required")
    dataset = AlleleDataset(frame, config)
    device = next(model.parameters()).device
    model.eval()
    values = {}
    for strand in ("forward", "reverse"):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                            collate_fn=AlleleCollator(tokenizer, config, strand), pin_memory=device.type == "cuda")
        values[strand] = np.concatenate([model(move_batch(batch, device))["tau"].float().cpu().numpy()
                                         for batch in loader])
    output = frame.copy()
    output["tau_forward"] = values["forward"]
    output["tau_reverse"] = values["reverse"]
    output["tau_raw"] = (values["forward"].astype(np.float64) + values["reverse"]) / 2
    if not np.isfinite(output.tau_raw).all():
        raise RuntimeError("Nonfinite model prediction")
    output["rho"] = output.assay.map(dict(zip(config.assays, config.rho)))
    output["tau"] = output.tau_raw
    if calibration is not None:
        if calibration["assays"] != config.assays or not np.array_equal(calibration["rho"], config.rho):
            raise ValueError("Calibration and checkpoint assay/dispersion identities differ")
        scales = np.asarray(calibration["scales"], float)
        if scales.shape != (len(config.assays),) or not np.isfinite(scales).all() or np.any((scales < 0) | (scales > 3)):
            raise ValueError("Invalid assay calibration scales")
        output["tau"] *= output.assay.map(dict(zip(config.assays, scales)))
    output["p_alt"] = expit(output.tau)
    return output
