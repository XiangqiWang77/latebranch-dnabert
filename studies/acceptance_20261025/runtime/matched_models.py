"""Matched focal readouts and depth-only interventions; no test-dependent choices."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from collator import AlleleCollator
from experiment import create_model as original_model, make_loader as original_loader, predict
from intervention_dnabert import ObservationDataset

DEPTHS = {f"depth{depth}": depth for depth in (0, 4, 8, 10)}
METHODS = {**DEPTHS, "vanilla_focal": None, "vanilla_attention": None}


class FocalCollator(AlleleCollator):
    """Keep the actual focal token separately for naturally tokenized REF and ALT."""

    def _vanilla_sequences(self, sequences, centers):
        center = len(sequences[0]) // 2
        edited = [seq[:center] + "ACGT"[int(base)] + seq[center + 1:]
                  for seq, base in zip(sequences, centers)]
        encoded = self._reference_sequences(edited)
        self._focal_encodings.append(encoded)
        return {key: encoded[key] for key in ("input_ids", "attention_mask")}

    def __call__(self, items):
        self._focal_encodings = []
        batch = super().__call__(items)
        if len(self._focal_encodings) != 2:
            raise RuntimeError("Expected exactly one REF and one ALT encoding")
        for allele, encoded in zip(("ref", "alt"), self._focal_encodings):
            for key in ("marker_position", "marker_span_length", "marker_offset", "was_truncated"):
                batch[f"{allele}_{key}"] = encoded[key]
            marker = encoded["marker_position"]
            if not torch.all(encoded["attention_mask"][torch.arange(len(marker)), marker] == 1):
                raise RuntimeError("Focal token was truncated or padded")
        self._focal_encodings = []
        return batch


class FocalVanilla(nn.Module):
    """Change only mean pooling to focal-token pooling in the existing Vanilla."""

    def __init__(self, original):
        super().__init__()
        self.backbone = original.backbone
        self.context_encoder = original.context_encoder
        self.potential = original.potential
        self.register_buffer("rho", original.rho.clone())

    def forward(self, batch):
        context = self.context_encoder(batch)
        scores = []
        inverse = batch["variant_inverse"]
        for allele in ("ref", "alt"):
            hidden = self.backbone(input_ids=batch[f"{allele}_input_ids"],
                                   attention_mask=batch[f"{allele}_attention_mask"])[0]
            marker = batch[f"{allele}_marker_position"]
            focal = hidden[torch.arange(len(hidden), device=hidden.device), marker][inverse]
            scores.append(self.potential(torch.cat([focal, context], dim=-1)).squeeze(-1))
        return {"tau": scores[1] - scores[0], "rho": self.rho[batch["assay_id"]]}


class AttentionVanilla(nn.Module):
    """Two natural sequence passes with LateBranch's exact FiLM and focal attention head."""

    def __init__(self, latebranch):
        super().__init__()
        self.backbone = latebranch.backbone
        self.context_encoder = latebranch.context_encoder
        self.terminal_film = latebranch.terminal_film
        self.head = latebranch.head
        self.register_buffer("rho", latebranch.rho.clone())

    def forward(self, batch):
        gamma, beta = self.terminal_film(self.context_encoder(batch)).chunk(2, dim=-1)
        inverse = batch["variant_inverse"]
        scores = []
        for allele in ("ref", "alt"):
            mask = batch[f"{allele}_attention_mask"]
            hidden = self.backbone(input_ids=batch[f"{allele}_input_ids"], attention_mask=mask)[0]
            hidden = hidden[inverse] * (1 + .1 * torch.tanh(gamma[:, None])) + beta[:, None]
            marker = batch[f"{allele}_marker_position"][inverse]
            focal = hidden[torch.arange(len(hidden), device=hidden.device), marker]
            attended, _ = self.head.local_attention(focal[:, None], hidden, hidden,
                                                    key_padding_mask=mask[inverse].eq(0),
                                                    need_weights=False)
            # The common potential offset cancels in ALT minus REF. Do not center
            # each single-candidate call, which would incorrectly return zero.
            scores.append(self.head.shared_scalar(torch.cat([focal, attended[:, 0]], dim=-1)).squeeze(-1))
        return {"tau": scores[1] - scores[0], "rho": self.rho[batch["assay_id"]]}


def create_model(method, frame, train_rows, rho, context_fields=("assay", "tissue")):
    if tuple(context_fields) != ("assay", "tissue"):
        raise ValueError("New matched controls exclude chromosome and donor inputs")
    if method in DEPTHS:
        # Construction at a common depth is essential: the legacy constructor
        # allocates depth-dependent temporary FiLM modules, consuming RNG draws.
        model, tokenizer = original_model("natural_offset_lb", frame, train_rows, rho, context_fields)
        model.branch_layer = DEPTHS[method]
        return model, tokenizer
    if method == "vanilla_focal":
        base, tokenizer = original_model("vanilla", frame, train_rows, rho, context_fields)
        return FocalVanilla(base), tokenizer
    if method == "vanilla_attention":
        base, tokenizer = original_model("natural_offset_lb", frame, train_rows, rho, context_fields)
        return AttentionVanilla(base), tokenizer
    raise ValueError(method)


def make_loader(method, frame, rows, sequences, tokenizer, batch_size, strand, shuffle=False, seed=13):
    if method in DEPTHS:
        return original_loader("natural_offset_lb", frame, rows, sequences, tokenizer,
                               batch_size, strand, shuffle, seed)
    if method not in ("vanilla_focal", "vanilla_attention"):
        raise ValueError(method)
    collator = FocalCollator(sequences, tokenizer, "vanilla", 128,
                            intervention_context="natural", strand=strand)
    return DataLoader(ObservationDataset(frame, rows), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collator, num_workers=0, pin_memory=True,
                      generator=torch.Generator().manual_seed(seed))
