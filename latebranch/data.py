"""Explicit CSV inputs and focal-preserving natural BPE tokenization."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

BASES = "ACGT"
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def read_table(path, counts=False):
    frame = pd.read_csv(path, dtype={"row_id": str, "group_id": str, "sequence": str,
                                    "ref": str, "alt": str, "assay": str, "tissue": str})
    required = {"row_id", "sequence", "ref", "alt", "assay", "tissue"}
    if counts:
        required |= {"n_ref", "n_alt"}
    missing = required - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Nonempty input required; missing columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any() or frame.row_id.duplicated().any():
        raise ValueError("Required fields must be nonmissing; row_id must be unique")
    for row in frame.itertuples(index=False):
        sequence = row.sequence
        if not isinstance(sequence, str) or len(sequence) < 3 or len(sequence) % 2 == 0:
            raise ValueError(f"Odd-length, centered REF window required at {row.row_id}")
        if (set(sequence) - set("ACGTN") or len(row.ref) != 1 or len(row.alt) != 1
                or row.ref not in BASES or row.alt not in BASES):
            raise ValueError(f"Use uppercase A/C/G/T alleles and A/C/G/T/N sequence at {row.row_id}")
        if sequence[len(sequence)//2] != row.ref:
            raise ValueError(f"REF does not match the central nucleotide at {row.row_id}")
    if {"n_ref", "n_alt"}.issubset(frame.columns):
        values = frame[["n_ref", "n_alt"]].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Counts must be finite, nonnegative integers")
        if (values.sum(axis=1) <= 0).any():
            raise ValueError("Total read count must be positive")
        frame[["n_ref", "n_alt"]] = values.astype(np.int64)
    return frame.reset_index(drop=True)


def check_partitions(frame):
    if not {"split", "group_id"}.issubset(frame.columns):
        raise ValueError("Training requires explicit split and group_id columns")
    if frame[["split", "group_id"]].isna().any().any():
        raise ValueError("Split and group identifiers cannot be missing")
    if set(frame.split) - {"train", "validation", "calibration", "test"}:
        raise ValueError("Split values: train, validation, calibration, test")
    if not {"train", "validation"}.issubset(set(frame.split)):
        raise ValueError("Training and validation partitions are required")
    if frame.groupby("group_id").split.nunique().max() > 1:
        raise ValueError("A group_id crosses partitions; use connected window/gene groups")
    # Identical windows should not cross partitions even when group labels differ.
    if frame.groupby("sequence").split.nunique().max() > 1:
        raise ValueError("An identical reference window crosses partitions")


class AlleleDataset(Dataset):
    def __init__(self, frame, config):
        self.records = frame.to_dict("records")
        assays = {name: i for i, name in enumerate(config.assays)}
        tissues = {name: i for i, name in enumerate(config.tissues)}
        for i, row in enumerate(self.records):
            if row["assay"] not in assays or row["tissue"] not in tissues:
                raise ValueError(f"Unknown assay/tissue at {row['row_id']}; the checkpoint vocabulary is fixed")
            row.update(row_index=i, assay_id=assays[row["assay"]], tissue_id=tissues[row["tissue"]],
                       ref_id=BASES.index(row["ref"]), alt_id=BASES.index(row["alt"]))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def encode_focal(sequences, tokenizer, max_tokens=128):
    encoded = tokenizer(sequences, add_special_tokens=False, return_offsets_mapping=True)
    rows, markers, spans, offsets, truncated = [], [], [], [], []
    available = max_tokens - 2
    for sequence, ids, positions in zip(sequences, encoded["input_ids"], encoded["offset_mapping"]):
        center = len(sequence)//2
        hits = [i for i, (start, end) in enumerate(positions) if start <= center < end]
        if len(hits) != 1:
            raise ValueError("The focal base must belong to exactly one natural BPE token")
        marker = hits[0]
        start, end = positions[marker]
        spans.append(end-start); offsets.append(center-start)
        truncated.append(len(ids) > available)
        if len(ids) > available:
            first = max(0, marker-available//2)
            last = min(len(ids), first+available)
            first = max(0, last-available)
            ids = ids[first:last]; marker -= first
        rows.append([tokenizer.cls_token_id, *ids, tokenizer.sep_token_id])
        markers.append(marker+1)
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.as_tensor(row); mask[i, :len(row)] = 1
    return dict(input_ids=ids, attention_mask=mask, marker_position=torch.tensor(markers),
                marker_span_length=torch.tensor(spans), marker_offset=torch.tensor(offsets),
                was_truncated=torch.tensor(truncated))


class AlleleCollator:
    def __init__(self, tokenizer, config, strand="forward", seed=13):
        if strand not in {"forward", "reverse", "random"}:
            raise ValueError("strand must be forward, reverse or random")
        self.tokenizer, self.config, self.strand = tokenizer, config, strand
        self.rng = np.random.default_rng(seed)

    def __call__(self, items):
        batch = {key: torch.tensor([row[key] for row in items], dtype=torch.long)
                 for key in ("row_index", "assay_id", "tissue_id", "ref_id", "alt_id")}
        # Retained solely for checkpoint-compatible context module shapes. It is
        # zeroed by the module; chromosome and donor are not model covariates.
        batch["chromosome_id"] = torch.zeros(len(items), dtype=torch.long)
        if all("n_alt" in row and "n_ref" in row for row in items):
            for key in ("n_ref", "n_alt"):
                batch[key] = torch.tensor([row[key] for row in items], dtype=torch.long)
            batch["total"] = batch["n_ref"] + batch["n_alt"]
        unique, inverse, lookup = [], [], {}
        for row in items:
            key = (row["sequence"], row["ref_id"], row["alt_id"])
            if key not in lookup:
                lookup[key] = len(unique); unique.append(key)
            inverse.append(lookup[key])
        flips = self.rng.random(len(unique)) < .5 if self.strand == "random" else np.full(len(unique), self.strand == "reverse")
        sequences = [sequence.translate(COMPLEMENT)[::-1] if flip else sequence
                     for (sequence, _, _), flip in zip(unique, flips)]
        refs = [3-ref if flip else ref for (_, ref, _), flip in zip(unique, flips)]
        alts = [3-alt if flip else alt for (_, _, alt), flip in zip(unique, flips)]
        batch["variant_inverse"] = torch.tensor(inverse)
        row_flips = torch.tensor(flips[inverse])
        batch["ref_id"][row_flips] = 3-batch["ref_id"][row_flips]
        batch["alt_id"][row_flips] = 3-batch["alt_id"][row_flips]
        if self.config.model_type == "latebranch":
            batch.update(encode_focal(sequences, self.tokenizer, self.config.max_tokens))
        else:
            for allele, bases in (("ref", refs), ("alt", alts)):
                edited = [seq[:len(seq)//2]+BASES[base]+seq[len(seq)//2+1:] for seq, base in zip(sequences,bases)]
                encoded = encode_focal(edited, self.tokenizer, self.config.max_tokens)
                batch.update({allele+"_"+key: value for key,value in encoded.items()})
        return batch
