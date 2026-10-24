# Third-party notices

## DNABERT-2

The backbone is [zhihan1996/DNABERT-2-117M](https://huggingface.co/zhihan1996/DNABERT-2-117M), revision `7bce263b15377fc15361f52cfab88f8b586abda0`. Its source, tokenizer/configuration and weight identities are recorded in [`backbone_manifest.json`](latebranch/backbone_manifest.json).

The upstream assets are distributed under Apache License 2.0. The unmodified upstream license is preserved in [`docs/DNABERT-2-LICENSE.txt`](docs/DNABERT-2-LICENSE.txt), and the download command retrieves the license alongside the model. DNABERT-2 acknowledges MosaicBERT as an implementation foundation. Its upstream code retains those notices.

The trained release contains adapted DNABERT-2 weights plus the project's added modules. Frozen pretrained weights are downloaded separately. No unrelated third-party model is vendored in this repository.

## Figures and report

The report's scientific-figure workflow used Scientific Agent Skills, cited there as Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065). This attribution does not imply endorsement or independent biological validation.

Python dependencies retain their respective licenses. A project-wide open-source license for the original LateBranch code has not been specified; the third-party license above is not a blanket license grant for all repository contents.
