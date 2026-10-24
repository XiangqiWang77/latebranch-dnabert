#!/usr/bin/env python3
"""Convert an existing processed study archive; this is not raw-read preprocessing.

The frame is a Python pickle: only use a local archive you trust. Components are
aligned to frame rows; nucleotide sequences are indexed by frame.variant_id.
"""
import argparse
import csv
from functools import lru_cache
import gzip
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


def export(frame_path, sequences_path, partitions_path, components_path, output, split='all'):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_pickle(frame_path)
    required = ['variant_id', 'ref_id', 'alt_id', 'assay_canonical', 'tissue', 'n_ref', 'n_alt']
    if not set(required).issubset(frame.columns) or frame[required].isna().any().any():
        raise ValueError(f'Nonmissing original frame columns required: {required}')
    sequences = np.load(sequences_path, mmap_mode='r', allow_pickle=False)
    components = np.load(components_path, allow_pickle=False)
    if components.ndim != 1 or len(components) != len(frame):
        raise ValueError('Components must align with observation rows, not sequence-array rows')
    if sequences.ndim != 2 or sequences.shape[1] < 3 or sequences.shape[1] % 2 != 1:
        raise ValueError('Sequences must be an odd-width 2D nucleotide array')
    with np.load(partitions_path, allow_pickle=False) as archive:
        parts = {('validation' if key == 'val' else key): archive[key] for key in archive.files}
    if not {'train', 'validation'}.issubset(parts) or set(parts) - {'train','validation','calibration','test'}:
        raise ValueError('Unexpected partition names or missing train/validation partitions')
    all_rows = np.concatenate(list(parts.values()))
    if (not np.issubdtype(all_rows.dtype, np.integer) or all_rows.ndim != 1
            or (all_rows < 0).any() or (all_rows >= len(frame)).any()
            or len(np.unique(all_rows)) != len(all_rows)):
        raise ValueError('Invalid or overlapping observation indices')
    seen_groups = set()
    for rows in parts.values():
        groups = set(components[rows].tolist())
        if seen_groups & groups:
            raise ValueError('A dependence component crosses partitions')
        seen_groups.update(groups)
    if split != 'all' and split not in parts:
        raise ValueError(f'No {split} partition in archive')
    bases = 'ACGTN'

    @lru_cache(maxsize=256)
    def decode(variant):
        if variant < 0 or variant >= len(sequences):
            raise ValueError(f'Invalid variant index: {variant}')
        row = sequences[variant]
        if (row < 0).any() or (row > 4).any() or not np.issubdtype(row.dtype, np.integer):
            raise ValueError('Nucleotide codes must be integers 0=A,1=C,2=G,3=T,4=N')
        return ''.join(bases[int(value)] for value in row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.conversion-', delete=False) as handle:
        temporary = Path(handle.name)
    count = 0
    try:
        opener = gzip.open if output.suffix == '.gz' else open
        with opener(temporary, 'wt', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['row_id','sequence','ref','alt','assay','tissue','n_ref','n_alt','split','group_id'])
            for label, rows in parts.items():
                if split != 'all' and label != split:
                    continue
                for index in rows:
                    row = frame.iloc[int(index)]
                    numeric = np.asarray([row.variant_id,row.ref_id,row.alt_id,row.n_ref,row.n_alt], float)
                    if not np.isfinite(numeric).all() or (numeric < 0).any() or (numeric != np.floor(numeric)).any():
                        raise ValueError(f'Invalid integer fields at observation {index}')
                    if row.ref_id > 3 or row.alt_id > 3 or row.n_ref + row.n_alt <= 0:
                        raise ValueError(f'Invalid allele identities or counts at observation {index}')
                    sequence = decode(int(row.variant_id))
                    ref, alt = bases[int(row.ref_id)], bases[int(row.alt_id)]
                    if sequence[len(sequence)//2] != ref:
                        raise ValueError(f'Central REF mismatch at observation {index}')
                    writer.writerow([int(index),sequence,ref,alt,row.assay_canonical,row.tissue,
                                     int(row.n_ref),int(row.n_alt),label,components[int(index)]])
                    count += 1
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', required=True, help='Trusted original frame.pkl.gz')
    parser.add_argument('--sequences', required=True, help='0..4 encoded reference-window .npy')
    parser.add_argument('--partitions', required=True, help='Observation-index .npz')
    parser.add_argument('--components', required=True, help='Observation-aligned component .npy')
    parser.add_argument('--output', required=True, help='New CSV or CSV.gz file')
    parser.add_argument('--split', choices=['all','train','validation','calibration','test'], default='all')
    args = parser.parse_args()
    count = export(args.frame,args.sequences,args.partitions,args.components,args.output,args.split)
    print(f'Exported {count} observations to {args.output}')


if __name__ == '__main__':
    main()
