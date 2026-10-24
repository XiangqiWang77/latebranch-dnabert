# Reproduction and evaluation

## Inference

The release `latebranch-window.pt` contains adapted parameters and the complete assay/tissue vocabulary. It is loaded over the pinned DNABERT-2 backbone. `configs/released_calibration.json` belongs to this exact checkpoint and assay order. Do not apply it to a newly trained model.

The public model was checked against the original frozen implementation on real held-out observations, on both strands, with exact token-geometry checks. It was also compared with archived FP32 predictions. See [`checkpoint_parity.json`](checkpoint_parity.json). This is a packaging-equivalence check, not another training replication.

The original paths and job scripts are intentionally not part of the public API. The model and count-likelihood operations originate from the files identified in [`source_provenance.json`](source_provenance.json). The input interface, CLI and portable checkpoint wrapper were added for this repository.

## New training

`latebranch train` reads only train/validation rows for fitting and checkpoint selection. Default optimization uses equal observation weights, float64 beta-binomial NLL, BF16 forward computation on CUDA, dropout disabled, and the last four backbone blocks adapted after a first epoch training the added modules. AdamW uses separate head/backbone learning rates. Validation averages the forward and reverse-complement logits. Checkpoints improve raw validation NLL by at least 1e-5; patience is 3 after at least 3 epochs, with an 8-epoch cap.

The portable data loader preserves the model's token geometry, strand and readout semantics. A new training run is not promised to reproduce an archived trajectory bit for bit across hardware, library versions or data ordering. The exact environment used for package verification is recorded separately.

## Calibration and evaluation

Prepare separate files for your calibration and test rows, retaining the `split` column. Generate raw predictions with your new checkpoint, then fit primary calibration scales on calibration data only:

```bash
latebranch predict --input data/calibration.csv.gz \
  --checkpoint runs/my-study/model.pt --backbone models/dnabert2 \
  --output runs/my-study/calibration_predictions.csv.gz

latebranch calibrate --predictions runs/my-study/calibration_predictions.csv.gz \
  --model-config runs/my-study/config.json \
  --output runs/my-study/calibration.json

latebranch predict --input data/test.csv.gz \
  --checkpoint runs/my-study/model.pt --backbone models/dnabert2 \
  --calibration runs/my-study/calibration.json \
  --output runs/my-study/test_predictions.csv.gz

latebranch evaluate --predictions runs/my-study/test_predictions.csv.gz \
  --output runs/my-study/metrics.csv
```

Primary calibration changes an assay's scale in [0,3] while retaining the training-fitted dispersion. The portable CLI does not automatically tune hyperparameters or compute sampling intervals. Its evaluation reports descriptive raw/calibrated NLL, null NLL, correlations and direction accuracy. Tied observed counts are excluded from direction accuracy; zero predicted direction receives half credit. Undefined correlations are left missing.

## Reproduce the archived evidence

The compact tables in [`results/`](../results/README.md) preserve all archived reference and matched-control outcomes. The report-source release archive includes compact predictions, the script regenerating Figure 2 and the numerical tables, source hashes and the original validation receipts. Figures 1, 3 and 4 are preserved exports. A separate figure/table bundle contains PDF/SVG/PNG files for direct use. The source archive's README records the XeLaTeX fonts needed for its Chinese text.

Those sampling intervals condition on fitted seed 13 checkpoints and fitted calibration. They are pointwise, not a retraining-seed analysis or a correction for the 69 historically inspected configurations. The window, chromosome and historical interpretation populations remain distinct. Original full-dataset training, raw-read processing and mapping-bias correction require additional source data; the supplied checkpoint and evidence capsule do not replace those inputs.
