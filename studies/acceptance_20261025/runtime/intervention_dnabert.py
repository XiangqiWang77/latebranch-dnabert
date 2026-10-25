"""Dataset/collator operations extracted unchanged from the study source.
Model primitives are those already verified in latebranch._core.
"""
from __future__ import annotations
from typing import Dict,Mapping,Sequence,List,Tuple,Optional
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from latebranch._core import BiologicalContextEncoder,AlignedFourBasePotentialHead,InterventionDNABERT,SiameseDNABERT,configure_backbone,effects_from_potentials
BASES="ACGT"
BASE_TO_ID={base:i for i,base in enumerate(BASES)}
COMPLEMENT=str.maketrans("ACGTN","TGCAN")
CANONICAL_ASSAYS=("CTCF","H3K27ac","H3K4me3","ATAC-seq","RNA-seq")
N_AUTOSOMES=22
class ObservationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, rows: np.ndarray):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.variant_id = frame["variant_id"].to_numpy(np.int32)
        self.tissue_id = frame["tissue_id"].to_numpy(np.int16)
        self.donor_id = frame["donor_id"].to_numpy(np.int8)
        self.assay_id = frame["assay_id"].to_numpy(np.int8)
        self.chromosome_id = frame["chromosome_id"].to_numpy(np.int8)
        self.ref_start = frame["ref_start"].to_numpy(np.int32)
        self.ref_id = frame["ref_id"].to_numpy(np.int8)
        self.alt_id = frame["alt_id"].to_numpy(np.int8)
        self.n_ref = frame["n_ref"].to_numpy(np.int32)
        self.n_alt = frame["n_alt"].to_numpy(np.int32)
        self.total = frame["total"].to_numpy(np.int32)
        self.label = frame["imbalance_significance"].to_numpy(np.uint8)
        self.sample_weight = frame["sample_weight"].to_numpy(np.float32)
        self.natural_train = (
            frame["natural_train"].astype(bool).to_numpy()
            if "natural_train" in frame
            else np.ones(len(frame), dtype=bool)
        )
        self.signal_train = (
            frame["signal_train"].astype(bool).to_numpy()
            if "signal_train" in frame
            else frame["imbalance_significance"].astype(bool).to_numpy()
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> Dict[str, int]:
        row = int(self.rows[item])
        return {
            "row_index": row,
            "variant_id": int(self.variant_id[row]),
            "tissue_id": int(self.tissue_id[row]),
            "donor_id": int(self.donor_id[row]),
            "assay_id": int(self.assay_id[row]),
            "chromosome_id": int(self.chromosome_id[row]),
            "ref_start": int(self.ref_start[row]),
            "ref_id": int(self.ref_id[row]),
            "alt_id": int(self.alt_id[row]),
            "n_ref": int(self.n_ref[row]),
            "n_alt": int(self.n_alt[row]),
            "total": int(self.total[row]),
            "label": int(self.label[row]),
            "sample_weight": float(self.sample_weight[row]),
            "natural_train": int(self.natural_train[row]),
            "signal_train": int(self.signal_train[row]),
        }

def decode_sequence(row: np.ndarray) -> str:
    return "".join(BASES[int(x)] if 0 <= int(x) < 4 else "N" for x in row)

def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]

class InterventionCollator:
    """Tokenize one shared context per unique variant in a minibatch."""

    def __init__(
        self,
        sequence_cache: np.ndarray,
        tokenizer,
        model_type: str,
        max_tokens: int,
        rc_augment: bool = False,
        intervention_context: str = "natural",
    ):
        self.sequence_cache = sequence_cache
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_tokens = max_tokens
        self.rc_augment = rc_augment
        # ``var`` and ``ref`` were accepted by an earlier, behaviorally inert
        # center-mode argument: both still produced the natural REF encoding.
        # Retain those aliases so old launch scripts preserve their exact input
        # semantics, while exposing the three actually distinct constructions.
        context_aliases = {"var": "natural", "ref": "natural"}
        intervention_context = context_aliases.get(
            intervention_context, intervention_context
        )
        if intervention_context not in {"natural", "ref_token", "mask"}:
            raise ValueError(
                f"Unknown intervention context {intervention_context!r}"
            )
        self.intervention_context = intervention_context
        self.marker_id = tokenizer.mask_token_id if tokenizer is not None else -1
        if tokenizer is not None and self.marker_id is None:
            raise RuntimeError("DNABERT tokenizer does not define [MASK] for the [VAR] marker.")

    @staticmethod
    def _relative_bucket(distance_bp: float) -> int:
        """Signed genomic-distance buckets centered on the intervention base."""

        if abs(distance_bp) < 0.5:
            return 7
        magnitude = abs(distance_bp)
        band = next(
            (index for index, boundary in enumerate((1, 2, 4, 8, 16, 32), start=1)
             if magnitude <= boundary),
            7,
        )
        return 7 + band if distance_bp > 0 else 7 - band

    def _aligned_context(self, sequences: Sequence[str]) -> Dict[str, torch.Tensor]:
        return self._aligned_with_center_tokens(
            sequences, [self.marker_id] * len(sequences), include_marker_position=True
        )

    def _aligned_with_center_tokens(
        self,
        sequences: Sequence[str],
        center_tokens: Sequence[int],
        include_marker_position: bool = False,
    ) -> Dict[str, torch.Tensor]:
        center = len(sequences[0]) // 2
        left = [seq[:center] for seq in sequences]
        right = [seq[center + 1 :] for seq in sequences]
        left_encoding = self.tokenizer(
            left, add_special_tokens=False, return_offsets_mapping=True
        )
        right_encoding = self.tokenizer(
            right, add_special_tokens=False, return_offsets_mapping=True
        )
        left_ids = left_encoding["input_ids"]
        right_ids = right_encoding["input_ids"]
        left_offsets = left_encoding["offset_mapping"]
        right_offsets = right_encoding["offset_mapping"]
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        available = self.max_tokens - 3
        encoded: List[List[int]] = []
        marker_positions: List[int] = []
        relative_buckets: List[List[int]] = []
        for lhs, rhs, lhs_offsets, rhs_offsets, center_token in zip(
            left_ids, right_ids, left_offsets, right_offsets, center_tokens
        ):
            lhs_distance = [0.5 * (start + end) - center for start, end in lhs_offsets]
            rhs_distance = [1.0 + 0.5 * (start + end) for start, end in rhs_offsets]
            if len(lhs) + len(rhs) > available:
                left_budget = min(len(lhs), available // 2)
                right_budget = min(len(rhs), available - left_budget)
                remaining = available - left_budget - right_budget
                add_left = min(remaining, len(lhs) - left_budget)
                left_budget += add_left
                remaining -= add_left
                right_budget += min(remaining, len(rhs) - right_budget)
                lhs = lhs[-left_budget:] if left_budget else []
                lhs_distance = lhs_distance[-left_budget:] if left_budget else []
                rhs = rhs[:right_budget]
                rhs_distance = rhs_distance[:right_budget]
            ids = [cls_id] + list(lhs) + [int(center_token)] + list(rhs) + [sep_id]
            marker_positions.append(1 + len(lhs))
            encoded.append(ids)
            relative_buckets.append(
                [15]
                + [self._relative_bucket(distance) for distance in lhs_distance]
                + [7]
                + [self._relative_bucket(distance) for distance in rhs_distance]
                + [15]
            )
        width = max(len(x) for x in encoded)
        input_ids = torch.full((len(encoded), width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(encoded), width), dtype=torch.long)
        position_bucket = torch.full((len(encoded), width), 15, dtype=torch.long)
        for i, (ids, buckets) in enumerate(zip(encoded, relative_buckets)):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[i, : len(ids)] = 1
            position_bucket[i, : len(buckets)] = torch.tensor(buckets, dtype=torch.long)
        result = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "relative_position_bucket": position_bucket,
        }
        if include_marker_position:
            result["marker_position"] = torch.tensor(marker_positions, dtype=torch.long)
            # Both explicit constructions insert exactly one center token for
            # exactly one nucleotide, unlike a natural BPE token which may span
            # several bases.
            result["marker_span_length"] = torch.ones(
                len(marker_positions), dtype=torch.long
            )
        return result

    def _full_sequences(self, sequences: Sequence[str], centers: Sequence[int]) -> Dict[str, torch.Tensor]:
        center_tokens = [int(self.tokenizer.convert_tokens_to_ids(BASES[int(base)])) for base in centers]
        return self._aligned_with_center_tokens(sequences, center_tokens)

    def _reference_token_sequences(
        self, sequences: Sequence[str], centers: Sequence[int]
    ) -> Dict[str, torch.Tensor]:
        """Insert the natural REF as one grounded token between tokenized flanks."""

        center_tokens = [
            int(self.tokenizer.convert_tokens_to_ids(BASES[int(base)]))
            for base in centers
        ]
        return self._aligned_with_center_tokens(
            sequences, center_tokens, include_marker_position=True
        )

    def _vanilla_sequences(
        self, sequences: Sequence[str], centers: Sequence[int]
    ) -> Dict[str, torch.Tensor]:
        center = len(sequences[0]) // 2
        edited = [
            sequence[:center] + BASES[int(base)] + sequence[center + 1 :]
            for sequence, base in zip(sequences, centers)
        ]
        # Both methods must retain the focal query under a short token budget.
        # Re-tokenize each complete allele naturally, then use the same
        # center-preserving crop as the natural-REF LateBranch input. Prefix
        # truncation can otherwise remove the only REF/ALT difference.
        encoded = self._reference_sequences(edited)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def _reference_sequences(self, sequences: Sequence[str]) -> Dict[str, torch.Tensor]:
        """Tokenize each unmodified REF sequence once and locate its center BPE token."""

        encoded = self.tokenizer(
            list(sequences),
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        all_ids = encoded["input_ids"]
        all_offsets = encoded["offset_mapping"]
        cls_id = int(self.tokenizer.cls_token_id)
        sep_id = int(self.tokenizer.sep_token_id)
        available = self.max_tokens - 2
        rows: List[List[int]] = []
        row_buckets: List[List[int]] = []
        marker_positions: List[int] = []
        marker_span_lengths: List[int] = []
        for sequence, token_ids, offsets in zip(sequences, all_ids, all_offsets):
            center = len(sequence) // 2
            marker = None
            for token, (start, end) in enumerate(offsets):
                if end > start and start <= center < end:
                    marker = token
            if marker is None:
                raise RuntimeError("Natural REF BPE tokenization lost the variant position")
            marker_start, marker_end = offsets[marker]
            marker_span_lengths.append(int(marker_end) - int(marker_start))
            if len(token_ids) > available:
                start_token = max(0, marker - available // 2)
                end_token = min(len(token_ids), start_token + available)
                start_token = max(0, end_token - available)
                token_ids = token_ids[start_token:end_token]
                offsets = offsets[start_token:end_token]
                marker -= start_token
            ids = [cls_id] + list(token_ids) + [sep_id]
            buckets = [15] + [
                self._relative_bucket(0.5 * (start + end) - center)
                for start, end in offsets
            ] + [15]
            marker += 1
            buckets[marker] = 7
            rows.append(ids)
            row_buckets.append(buckets)
            marker_positions.append(marker)
        width = max(len(row) for row in rows)
        input_ids = torch.full(
            (len(rows), width), int(self.tokenizer.pad_token_id), dtype=torch.long
        )
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        relative_bucket = torch.full((len(rows), width), 15, dtype=torch.long)
        for row, (ids, buckets) in enumerate(zip(rows, row_buckets)):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
            relative_bucket[row, : len(buckets)] = torch.tensor(buckets, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "marker_position": torch.tensor(marker_positions, dtype=torch.long),
            "marker_span_length": torch.tensor(
                marker_span_lengths, dtype=torch.long
            ),
            "relative_position_bucket": relative_bucket,
        }

    def __call__(self, items: Sequence[Mapping[str, int]]) -> Dict[str, torch.Tensor]:
        fields = (
            "row_index",
            "variant_id",
            "tissue_id",
            "donor_id",
            "assay_id",
            "chromosome_id",
            "ref_start",
            "ref_id",
            "alt_id",
            "n_ref",
            "n_alt",
            "total",
            "label",
            "natural_train",
            "signal_train",
        )
        batch = {name: torch.tensor([int(x[name]) for x in items], dtype=torch.long) for name in fields}
        batch["sample_weight"] = torch.tensor(
            [float(x["sample_weight"]) for x in items], dtype=torch.float32
        )
        if self.model_type == "metadata":
            return batch
        vids = batch["variant_id"].numpy()
        unique_vids, inverse = np.unique(vids, return_inverse=True)
        sequences = [decode_sequence(self.sequence_cache[int(v)]) for v in unique_vids]
        batch["variant_inverse"] = torch.tensor(inverse, dtype=torch.long)
        if self.rc_augment:
            flip_unique = np.random.random(len(unique_vids)) < 0.5
            sequences = [reverse_complement(seq) if flip else seq for seq, flip in zip(sequences, flip_unique)]
            flip_rows = torch.from_numpy(flip_unique[inverse])
            # With A,C,G,T encoded as 0,1,2,3, complement(base) = 3-base.
            batch["ref_id"][flip_rows] = 3 - batch["ref_id"][flip_rows]
            batch["alt_id"][flip_rows] = 3 - batch["alt_id"][flip_rows]
        unique_ref = []
        unique_alt = []
        first_by_vid = {int(v): i for i, v in enumerate(vids)}
        for vid in unique_vids:
            j = first_by_vid[int(vid)]
            unique_ref.append(int(batch["ref_id"][j]))
            unique_alt.append(int(batch["alt_id"][j]))
        if self.model_type == "intervention":
            if self.intervention_context == "natural":
                context = self._reference_sequences(sequences)
            elif self.intervention_context == "ref_token":
                context = self._reference_token_sequences(sequences, unique_ref)
            else:
                context = self._aligned_context(sequences)
            batch.update(context)
        elif self.model_type in {"siamese", "vanilla"}:
            encode = self._vanilla_sequences if self.model_type == "vanilla" else self._full_sequences
            ref = encode(sequences, unique_ref)
            alt = encode(sequences, unique_alt)
            batch.update(
                {
                    "ref_input_ids": ref["input_ids"],
                    "ref_attention_mask": ref["attention_mask"],
                    "alt_input_ids": alt["input_ids"],
                    "alt_attention_mask": alt["attention_mask"],
                }
            )
        else:
            raise ValueError(f"Unknown model type {self.model_type!r}")
        return batch

class MetadataOnly(nn.Module):
    def __init__(
        self,
        n_tissues: int,
        n_assays: int,
        rho: np.ndarray,
        context_fields: Sequence[str] = ("assay", "tissue", "chromosome"),
        context_pairwise: bool = True,
    ):
        super().__init__()
        width = 64
        self.base = nn.Embedding(4, width)
        self.context_encoder = BiologicalContextEncoder(
            n_tissues=n_tissues,
            n_assays=n_assays,
            hidden=width,
            width=width,
            fields=context_fields,
            use_pairwise_interactions=context_pairwise,
        )
        self.potential = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.register_buffer("rho", torch.tensor(rho, dtype=torch.float32))

    def score(self, base: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.potential(
            torch.cat([self.base(base), context], dim=-1)
        ).squeeze(-1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        context = self.context_encoder(batch)
        ref = self.score(batch["ref_id"], context)
        alt = self.score(batch["alt_id"], context)
        return {"tau": alt - ref, "rho": self.rho[batch["assay_id"]]}

def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
