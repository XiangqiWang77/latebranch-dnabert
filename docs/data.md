# Data format

Each CSV/CSV.gz row describes one assay/tissue observation for one queried SNV.

| Column | Required for | Definition |
|---|---|---|
| `row_id` | All commands | Unique observation identifier |
| `sequence` | Prediction/training | Uppercase odd-length natural REF window; the SNV is the central nucleotide |
| `ref`, `alt` | Prediction/training | Single A/C/G/T nucleotides; `ref` must match the sequence center |
| `assay`, `tissue` | Prediction/training | Condition names from the checkpoint's vocabulary, or training vocabulary for a new fit |
| `n_ref`, `n_alt` | Training/evaluation/calibration | Nonnegative integer counts with positive total |
| `split` | Training/calibration | `train`, `validation`, `calibration`, or `test` |
| `group_id` | Training | Dependence component; no group may cross partitions |

The released study used 257-bp windows. N is allowed in the flanking sequence; REF and ALT must be unambiguous A/C/G/T bases. The BPE token budget defaults to 128 and preserves the focal token when cropping. Both strand views use their own exact token offsets and spans.

The released checkpoint recognizes 5 assays and 31 tissues, listed in [`released_model.json`](../configs/released_model.json). Unknown conditions are rejected rather than silently mapped to a known tissue. REF=ALT is supported as an identity control and yields zero log ratio.

`group_id` must reflect the dependence structure appropriate to your data. For the original study this starts with connected overlapping sequence windows; gene/window grouping is used in a sensitivity analysis. The input validator checks the group labels you supply and identical-window overlap. It does not independently reconstruct genomic overlap, shared reads, haplotypes or gene dependence from the CSV.

The [example variant queries](../examples/variants.csv) and [count table](../examples/counts.csv) are synthetic formatting examples. The small count example is not sufficient for fitting the default training-only dispersions, which require at least 30 observations per assay. Full processed EN-TEx training data and raw read files are not embedded in this repository.

If you have the original processed archive, [`export_study_data.py`](../scripts/export_study_data.py) converts its count frame, nucleotide array, partition indices and component identifiers into this format. The helper requires explicit input paths; it has no cluster-specific defaults. The published data source is [EN-TEx](https://doi.org/10.1016/j.cell.2023.02.018). Downloading that source alone does not reconstruct this study's complete QC and partition history.

```bash
python scripts/export_study_data.py \
  --frame archive/frame.pkl.gz \
  --sequences archive/ref_context_r128.u8.npy \
  --partitions archive/window.npz \
  --components archive/components.npy \
  --output data/observations.csv.gz
```

The frame is a trusted local Python pickle. Nucleotide codes are 0=A, 1=C, 2=G, 3=T, 4=N. Component labels align with observation-frame rows; sequence-array rows are indexed by `variant_id`. The converter checks row/group separation and central REF identity. Add `--split calibration` or `--split test` to export the separate files used by the calibration workflow.
