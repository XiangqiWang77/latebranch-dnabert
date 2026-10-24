# Method

## 1. From pretrained tokens to an allelic target

DNABERT-2's token-prediction objective learns genomic representations. A token probability does not directly specify an assay- and tissue-specific molecular ALT fraction. The target here is

$$\eta=\log(\mu_{ALT}/\mu_{REF}),\qquad p_{ALT}=\sigma(\eta).$$

A supervised two-pass DNABERT model can learn this target. LateBranch studies a different interface for introducing allele identity while keeping reference coordinates explicit.

## 2. An intervention-inspired interface

The natural REF sequence is encoded through a shared prefix. At the token containing the central SNV, candidate base `b` receives a residual relative to reference base `r`:

$$\delta_b=gP(e_b-e_r)\odot[1+0.5\tanh(W\phi)],$$

$$\phi=(o/\ell,(\ell-1-o)/\ell,1/\ell,\log(\ell)/5).$$

Here `o` is the zero-based position inside a token and `ell` is its span. The position gate starts at zero; the REF residual is exactly zero. Candidate paths share upper-layer weights, terminal assay/tissue FiLM and a focal-query attention readout. A shared scalar head produces four potentials; ALT minus REF defines the output log ratio.

The design borrows the controlled-state-change perspective of intervention-based interpretation. Candidate states retain REF token coordinates; they are learned internal alternatives, not exact retokenizations of a natural ALT sequence. Inspectability of this computation does not establish a biological causal mechanism.

## 3. Learning and interpreting the output

The training objective is beta-binomial count NLL with a shared, training-fitted dispersion for each assay. Forward and reverse-complement logits are averaged at inference. Optional assay scales are fitted on separate calibration observations. Raw and calibrated effects remain separate in every output.

The released model uses depth 8 and explicit within-token coordinates. The repository does not claim that depth 8 is optimal or that the position gate independently improves prediction. Complete readout/depth/position comparisons are retained in [the result tables](../results/README.md).

The [report](../reports/positive.pdf) includes ATCG sequence logos, focal-seqlet coverage and attention-distance analysis from an earlier balanced experiment. GC-rich and A-rich examples describe model sequence sensitivities. Motif-catalog similarities and attention overlap are hypothesis-generating, not TF identification or biological intervention evidence.
