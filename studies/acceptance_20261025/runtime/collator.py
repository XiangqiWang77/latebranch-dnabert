"""Explicit focal offset, strand views, and reference-independent masked context."""
import common
import numpy as np
import torch
from intervention_dnabert import InterventionCollator, decode_sequence, reverse_complement


class AlleleCollator(InterventionCollator):
    def __init__(self, *args, strand="random", **kwargs):
        super().__init__(*args, rc_augment=False, **kwargs)
        if strand not in {"forward", "reverse", "random"}:
            raise ValueError(strand)
        self.strand = strand

    def _reference_sequences(self, sequences):
        result = super()._reference_sequences(sequences)
        offsets = self.tokenizer(list(sequences), add_special_tokens=False,
                                 return_offsets_mapping=True)["offset_mapping"]
        positions, truncated = [], []
        for seq, spans in zip(sequences, offsets):
            center = len(seq) // 2
            hits = [(start, end) for start, end in spans if start <= center < end]
            if len(hits) != 1:
                raise RuntimeError("Focal nucleotide must belong to exactly one token")
            positions.append(center - hits[0][0])
            truncated.append(len(spans) + 2 > self.max_tokens)
        result["marker_offset"] = torch.tensor(positions, dtype=torch.long)
        result["was_truncated"] = torch.tensor(truncated, dtype=torch.bool)
        return result

    def _aligned_with_center_tokens(self, sequences, center_tokens, include_marker_position=False):
        result = super()._aligned_with_center_tokens(sequences, center_tokens, include_marker_position)
        if include_marker_position:
            result["marker_offset"] = torch.zeros(len(sequences), dtype=torch.long)
            center = len(sequences[0]) // 2
            left = self.tokenizer([s[:center] for s in sequences], add_special_tokens=False)["input_ids"]
            right = self.tokenizer([s[center + 1:] for s in sequences], add_special_tokens=False)["input_ids"]
            result["was_truncated"] = torch.tensor([
                len(a) + len(b) + 3 > self.max_tokens for a, b in zip(left, right)], dtype=torch.bool)
        return result

    def __call__(self, items):
        fields = ("row_index", "variant_id", "tissue_id", "donor_id", "assay_id",
                  "chromosome_id", "ref_start", "ref_id", "alt_id", "n_ref", "n_alt",
                  "total", "label", "natural_train", "signal_train")
        batch = {key: torch.tensor([int(x[key]) for x in items], dtype=torch.long) for key in fields}
        batch["sample_weight"] = torch.ones(len(items))
        # Deduplicate by both variant and queried allele, never assume unique ALT per locus.
        keys = np.column_stack([batch["variant_id"].numpy(), batch["ref_id"].numpy(), batch["alt_id"].numpy()])
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        seqs = [decode_sequence(self.sequence_cache[int(key[0])]) for key in unique]
        if self.strand == "random":
            flips = np.random.random(len(unique)) < .5
        else:
            flips = np.full(len(unique), self.strand == "reverse", dtype=bool)
        seqs = [reverse_complement(seq) if flip else seq for seq, flip in zip(seqs, flips)]
        row_flips = torch.from_numpy(flips[inverse])
        batch["ref_id"][row_flips] = 3 - batch["ref_id"][row_flips]
        batch["alt_id"][row_flips] = 3 - batch["alt_id"][row_flips]
        batch["variant_inverse"] = torch.tensor(inverse, dtype=torch.long)
        unique_ref = np.where(flips, 3 - unique[:, 1], unique[:, 1]).tolist()
        unique_alt = np.where(flips, 3 - unique[:, 2], unique[:, 2]).tolist()
        if self.model_type == "metadata":
            return batch
        if self.model_type == "intervention":
            if self.intervention_context == "natural":
                seqs = [seq[:len(seq)//2] + "ACGT"[ref] + seq[len(seq)//2+1:] for seq, ref in zip(seqs, unique_ref)]
                batch.update(self._reference_sequences(seqs))
            elif self.intervention_context == "mask":
                batch.update(self._aligned_context(seqs))
            else:
                raise ValueError(self.intervention_context)
        elif self.model_type == "vanilla":
            ref = self._vanilla_sequences(seqs, unique_ref)
            alt = self._vanilla_sequences(seqs, unique_alt)
            batch.update(ref_input_ids=ref["input_ids"], ref_attention_mask=ref["attention_mask"],
                         alt_input_ids=alt["input_ids"], alt_attention_mask=alt["attention_mask"])
        else:
            raise ValueError(self.model_type)
        return batch
