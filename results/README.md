# Results

Selected tables from the fixed-seed study cover predictive quality, query cost and model ablations. The reference comparison uses two-pass Vanilla DNABERT with a mean-pool readout; the matched controls separate readout, injection-depth and position-gate choices.

| Evidence | Population and contents |
|---|---|
| [Quality, latency and memory](../studies/acceptance_20261025/results/quality_cost_memory.csv) | Seven benchmarked checkpoints, across both splits and all four query workloads |
| [Data and checkpoint replay](../studies/acceptance_20261025/README.md) | Real inputs, configurations, calibrations and reconstruction commands |
| [`reference/metrics.csv`](reference/metrics.csv) | Eight fitted models across two splits: 97,662 window-test observations and 249,984 chromosome-test observations; raw, primary scale calibration and joint-calibration sensitivity |
| [`reference/paired_intervals.csv`](reference/paired_intervals.csv) | Paired NLL contrasts for the reference experiment |
| [`controls/pooled_comparison.csv`](controls/pooled_comparison.csv) | Matched-analysis models/constants, pooled raw and calibrated endpoints; a convenience subset of `controls/metrics.csv` |
| [`controls/metrics.csv`](controls/metrics.csv) | Matched-analysis outcomes and predefined strata on the 97,662 window-test observations |
| [`controls/paired_intervals.csv`](controls/paired_intervals.csv) | Retained paired contrasts, strata and dependence units |
| [`patterns/`](patterns/) | Earlier balanced experiment: focal-seqlet coverage and paired intervals for 1,280 outcome-stratified cases |

Reliability bins and fitted null constants are retained beside their corresponding tables. Missing values remain missing. The source hashes and destinations of copied artifacts are recorded in [`artifact_manifest.json`](artifact_manifest.json).

## Reading model and endpoint names

`natural_offset_lb` and matched-analysis `depth8` identify the released position-aware LateBranch architecture. `vanilla` is the original two-pass mean-pool reference. In the subsequent controls, `vanilla_mean`, `vanilla_focal` and `vanilla_attention` distinguish readouts, while `position_off` removes within-token gating. `depth0`, `depth4` and `depth10` change the injection depth. Model options supported by the public training interface are described in [`configs/train.json`](../configs/train.json); the compact result archive also preserves additional study comparators that are not reimplemented in this package.

NLL is in nats per observation; smaller is better. A paired `difference` is left minus right, as identified by its columns (reference tables use the named LateBranch-versus-comparator contrast). Correlations use the smoothed observed log count ratio. Direction accuracy excludes tied observed counts and gives half credit to zero predictions. Calibration changes magnitudes without providing an independent model fit.

## Verification and scope

The published tables contain a selection of study models. Numerical results and intervals for the retained models are unchanged.

The estimates and pointwise intervals condition on seed 13 fits and fitted calibration. Historical model inspection, comparison multiplicity, cross-split population differences and the separate interpretation cohort still govern their interpretation. These tables support endpoint-specific comparisons; they do not establish an optimal injection depth, a benefit from the position gate, universal superiority or external transfer. The original reference comparison and later matched controls remain separately labeled.
