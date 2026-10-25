"""Explicit paths for the frozen study replay; no cluster configuration required."""
import json,os
from pathlib import Path
import numpy as np
import pandas as pd

DATA=Path(os.environ['LATEBRANCH_STUDY_DATA'])
MODEL=Path(os.environ['LATEBRANCH_BACKBONE'])
ASSAYS=['CTCF','H3K27ac','H3K4me3','ATAC-seq','RNA-seq']

def load_data(split):
    frame=pd.read_csv(DATA/'observations.csv.gz')
    if not np.array_equal(frame.row_index,np.arange(len(frame))):raise ValueError('Observation order changed')
    frame['source_variant_id']=frame.variant_id
    frame['variant_id']=frame.sequence_id
    frame['sample_weight']=np.float32(1);frame['natural_train']=np.uint8(1)
    frame['signal_train']=frame.imbalance_significance.astype(np.uint8)
    with np.load(DATA/f'{split}.npz') as z:parts={k:z[k] for k in z.files}
    sequences=np.load(DATA/'ref_sequences.npy',mmap_mode='r')
    rho=np.asarray(json.loads((DATA/f'{split}.json').read_text())['shared_rho'])
    return frame,parts,sequences,rho
