# Study data and checkpoints

Real sequence/count inputs, fixed data partitions, model definitions and portable reconstruction for the released study models.

## Download

Install the package using the [main instructions](../../README.md), then download the [v0.2.0 assets](https://github.com/XiangqiWang77/latebranch-dnabert/releases/tag/v0.2.0):

```bash
mkdir -p downloads
gh release download v0.2.0 --repo XiangqiWang77/latebranch-dnabert \
  --pattern 'processed-study-inputs.zip' --pattern '*-comparators.zip' \
  --dir downloads
python studies/acceptance_20261025/verify_assets.py --directory downloads
python -m zipfile -e downloads/processed-study-inputs.zip downloads
python -m zipfile -e downloads/retained-window-comparators.zip downloads
python -m zipfile -e downloads/retained-chromosome-comparators.zip downloads
latebranch download-backbone --output models/dnabert2
```

| Asset | Contents |
|---|---|
| `processed-study-inputs.zip` | 971,819 observations, 416,632 real 257-bp REF windows, counts, metadata and train/validation/calibration/test partitions |
| `retained-window-comparators.zip` | Four window-model checkpoints with configurations, training histories, selection records, calibrations and prediction arrays |
| `retained-chromosome-comparators.zip` | Three chromosome-model checkpoints with the same supporting files |

The [asset manifest](asset_manifest.json) records file sizes, download URLs and SHA-256 values. The backbone is fetched separately at a fixed revision.

In `processed/`, the `sequence_id` column of `observations.csv.gz` indexes `ref_sequences.npy` (0=A, 1=C, 2=G, 3=T). `variant_ids.npy` preserves original variant IDs. `window.npz` and `chromosome.npz` contain partition row indices; their JSON files contain training-fitted assay dispersions. Metadata and per-file hashes are included in the archive.

## Replay a checkpoint

```bash
python studies/acceptance_20261025/replay.py \
  --inputs downloads/processed --bundle downloads/models \
  --backbone models/dnabert2 --model window__natural_offset_lb \
  --phase test --limit 64 --device cpu --output runs/latebranch-replay.npz
```

Choose a model key from [registry.json](registry.json). Supported phases are `validation`, `calibration` and `test`; `--limit 0` evaluates a full partition. Use `--device cuda` for GPU inference. The command verifies the backbone and model weights, then compares raw predictions with the corresponding archived arrays. Calibration parameters are supplied separately in each model directory.

All seven models passed reconstruction and archived-prediction comparison on four real held-out observations each. [Validation records](asset_validation.json) identify the tested runtime and numerical errors. This checks packaged-model reconstruction; full training reproduction is a separate operation.

## Model definitions

[model_definitions.json](model_definitions.json) records architecture, conditioning, optimization and selection for every checkpoint. All models use assay/tissue conditioning, validation NLL checkpoint selection and a separate calibration partition. Fits use up to eight epochs and count-likelihood objectives.

The maintained training interface is described in [Reproduction](../../docs/reproduction.md). The study replay scripts are included in this directory.

## Result tables

| File | Contents |
|---|---|
| [Predictive quality](results/predictive_quality.csv) | Raw/calibrated full-test NLL for all seven seed-13 checkpoints |
| [Quality, latency and memory](results/quality_cost_memory.csv) | All models, both splits, batch sizes 1/32, one/three ALT queries, and GPU-only/end-to-end timing |
| [Individual timings](results/benchmark_repeats.csv) | All 1,120 retained technical timing measurements |
| [Forward-equivalence checks](results/benchmark_checks.json) | 28 optimized-query comparisons with ordinary prediction |
| [Benchmark receipt](results/benchmark_complete.json) | Hardware, precision, source and model identities |
| [Quality–cost figure](figures/quality_cost.pdf) | All model/workload comparisons; left column window, right column chromosome; rows batch-1/one-ALT, batch-32/one-ALT and batch-32/three-ALT |
| [Memory figure](figures/memory.pdf) | Peak allocated memory for batch-32, three-ALT queries; A window, B chromosome |

NLL is in nats per observation. Timing uses the same GPU allocation, FP32 with TF32 disabled, both strand views, and REF reuse for multi-ALT queries. End-to-end timing includes tokenization, transfer and forward evaluation; model loading and disk I/O are excluded. Timing intervals describe technical repeats. These fixed-seed measurements do not estimate training variability.

In the quality–cost figure, blue circles denote LateBranch; green triangles neutral-terminal; gray diamonds Vanilla mean; and pink downward triangles matched two-pass attention. Horizontal intervals are the interquartile range of 20 timing repeats. The chromosome comparison has no matched-attention fit.
