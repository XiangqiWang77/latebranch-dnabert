"""Shared model construction, strand averaging, and leakage-free loaders."""
from common import MODEL
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from collator import AlleleCollator
from model import LocalizedAlleleDNABERT
from intervention_dnabert import ObservationDataset, MetadataOnly, SiameseDNABERT, configure_backbone, move_batch

METHODS = {
    "natural_lb": dict(model_type="intervention", context="natural", branch=8, offset=False, neutral=False),
    "natural_offset_lb": dict(model_type="intervention", context="natural", branch=8, offset=True, neutral=False),
    "neutral_lb": dict(model_type="intervention", context="mask", branch=8, offset=False, neutral=True),
    "neutral_terminal": dict(model_type="intervention", context="mask", branch=12, offset=False, neutral=True),
    "vanilla": dict(model_type="vanilla", context="natural"),
    "metadata": dict(model_type="metadata", context="natural"),
}


def create_model(method, frame, train_rows, rho, context_fields=("assay", "tissue")):
    spec = METHODS[method]
    n_tissues = int(frame.tissue_id.max()) + 1
    if method == "metadata":
        model = MetadataOnly(n_tissues, 5, rho, context_fields=context_fields)
        model.rho = torch.as_tensor(rho, dtype=torch.float64)
        return model, None
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    backbone = AutoModel.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    configure_backbone(backbone, freeze=False, train_last_n=4)
    if method == "vanilla":
        model = SiameseDNABERT(backbone, n_tissues, 5, rho, context_fields=context_fields)
    else:
        base_ids = [tokenizer.convert_tokens_to_ids(base) for base in "ACGT"]
        for base, base_id in zip("ACGT", base_ids):
            if tokenizer(base, add_special_tokens=False)["input_ids"] != [base_id]:
                raise RuntimeError(f"Invalid single-base grounding: {base}")
        initial = backbone.get_input_embeddings().weight[base_ids].detach().clone()
        model = LocalizedAlleleDNABERT(backbone, n_tissues, 5, rho, base_initialization=initial,
            branch_layer=spec["branch"], context_fields=context_fields,
            neutral_anchor=spec["neutral"], position_aware=spec["offset"])
    # Allele contrasts must not be dominated by independent dropout masks.
    # Apply the same deterministic-dropout policy to all DNABERT competitors.
    for layer in model.modules():
        if isinstance(layer, nn.Dropout):
            layer.p = 0.
    model.rho = torch.as_tensor(rho, dtype=torch.float64)
    return model, tokenizer


def make_loader(method, frame, rows, sequences, tokenizer, batch_size, strand, shuffle=False, seed=13):
    spec = METHODS[method]
    collator = AlleleCollator(sequences, tokenizer, spec["model_type"], 128,
                             intervention_context=spec["context"], strand=strand)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ObservationDataset(frame, rows), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collator, num_workers=0, pin_memory=True, generator=generator)


@torch.no_grad()
def predict(model, loaders, device, block_nonfocal=False):
    model.eval()
    output = {}
    for strand, loader in loaders.items():
        rows, values = [], []
        for raw in loader:
            batch = move_batch(raw, device)
            result = model(batch, block_nonfocal=True) if block_nonfocal else model(batch)
            rows.append(batch["row_index"].cpu().numpy())
            values.append(result["tau"].float().cpu().numpy())
        indexes = np.concatenate(rows)
        if "row_index" in output:
            assert np.array_equal(indexes, output["row_index"])
        output["row_index"] = indexes
        output["tau_" + strand] = np.concatenate(values)
    output["tau"] = (output["tau_forward"].astype(np.float64) + output["tau_reverse"]) / 2
    if not np.isfinite(output["tau"]).all():
        raise RuntimeError("Nonfinite model predictions")
    return output
