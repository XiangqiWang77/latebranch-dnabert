# LateBranch-DNABERT

**Shared reference encoding and localized allele interventions for allelic log-ratio prediction.**

LateBranch adapts DNABERT-2 to predict assay- and tissue-specific allelic imbalance from DNA sequence. It encodes a natural reference window once, introduces A/C/G/T candidate updates at the variant's token, and propagates those candidates through shared upper layers. An attention readout assigns candidate potentials; the ALT-minus-REF difference predicts the log allelic ratio and ALT fraction.

<img src="figures/overview.png" alt="LateBranch architecture: shared reference encoding, localized candidate updates, within-token position features and an allele-specific attention readout" width="900">

[Method](docs/method.md) · [Input format](docs/data.md) · [Reproduction](docs/reproduction.md) · [Results](results/README.md) · [Figures](figures/README.md)

## Results against Vanilla DNABERT

The reference comparison uses a supervised two-pass DNABERT-2 model with a mean-pool readout. The table reports calibrated count NLL on the full test partitions for the fixed seed-13 fits; lower is better.

| Test split | Observations | Vanilla | LateBranch |
|---|---:|---:|---:|
| Window | 97,662 | 2.771044 | 2.765614 |
| Chromosome 19–22 | 249,984 | 2.720008 | 2.717552 |

For batch-32 queries with three ALT alleles per window, LateBranch takes **83.82 ms** versus **173.30 ms** for Vanilla, a **2.07× speedup**. Both models reuse REF computation, and timing includes sequence encoding, transfer and forward/reverse-complement prediction on the same hardware and precision. This speed ratio describes the specified multi-ALT workload.

[Complete results](results/README.md) include raw and calibrated metrics, paired uncertainty intervals, all comparator models, other query workloads and memory measurements. Sequence-pattern figures identify their model and cohort in the [figure captions](figures/README.md).

## Installation

Use Python 3.10 and a PyTorch installation suitable for your CPU or CUDA environment. The package pins the DNABERT-compatible Transformers and tokenizer versions.

```bash
git clone https://github.com/XiangqiWang77/latebranch-dnabert.git
cd latebranch-dnabert
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Download and predict

Download the pinned DNABERT-2 backbone and the trained LateBranch weights:

```bash
latebranch download-backbone --output models/dnabert2
python scripts/download_checkpoint.py --output models

latebranch predict \
  --input examples/variants.csv \
  --checkpoint models/latebranch-window.pt \
  --backbone models/dnabert2 \
  --calibration configs/released_calibration.json \
  --output runs/example/predictions.csv \
  --device cpu
```

The example contains synthetic DNA queries, including a REF=ALT identity control. [Expected predictions](examples/expected_predictions.csv) are provided. Use `--device cuda` for GPU inference.

| Output | Meaning |
|---|---|
| `tau_raw` | Mean forward and reverse-complement logit |
| `tau` | Calibrated log ALT/REF; equals `tau_raw` without calibration |
| `p_alt` | Predicted ALT fraction, `sigmoid(tau)` |
| `rho` | Training-fitted assay dispersion for the count likelihood |

Input assay and tissue labels must match the [checkpoint vocabulary](configs/released_model.json). See the [input specification](docs/data.md) for sequence, allele and count fields.

## Train and evaluate

Provide explicit train/validation partitions and dependence-group identifiers:

```bash
latebranch check-data --input data/observations.csv.gz --training
latebranch train \
  --input data/observations.csv.gz \
  --config configs/train.json \
  --backbone models/dnabert2 \
  --output runs/my-study \
  --device cuda
```

Default training uses count likelihood, one epoch adapting the added modules, then adaptation of the last four backbone blocks, with validation NLL checkpoint selection. Set `model_type` to `vanilla` for the two-pass reference. [Reproduction instructions](docs/reproduction.md) cover training settings, separate calibration and held-out evaluation.

The [study data and checkpoints](studies/acceptance_20261025/README.md) provide real sequence/count inputs, fixed partitions and a portable replay command for the released models. Run the lightweight CPU checks with `python -m unittest discover -s tests -v`.

## Citation and attribution

LateBranch builds on [DNABERT-2](https://arxiv.org/abs/2306.15006) and uses the [EN-TEx resource](https://doi.org/10.1016/j.cell.2023.02.018). Intervention-based interpretation motivates the candidate-state interface; see [Geiger et al.](https://papers.nips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html).

See [CITATION.cff](CITATION.cff) for the software citation and [third-party notices](THIRD_PARTY_NOTICES.md) for dependency attribution. The downloaded DNABERT-2 assets retain their Apache-2.0 license; a project-wide license for the original LateBranch code has not been specified.
