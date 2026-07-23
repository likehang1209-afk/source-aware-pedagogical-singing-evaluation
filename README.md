# Source-Aware Evaluation of Pedagogical Singing Attributes

This repository is the reproducibility package for:

> **Source-Aware Evaluation of Pedagogical Singing Attribute Recognition:
> Class-Imbalance Tradeoffs and Acoustic Evidence Auditing**

The study evaluates seven SVQTD pedagogical singing attributes under nested
leave-one-aria-out testing with recording-source-disjoint inner development.
The contribution is an evaluation and evidence-audit framework rather than a
new foundation-model architecture.

## What Is Included

- Duplicate-resolution and cohort-audit code.
- Six held-aria outer folds and five source-disjoint inner splits.
- Pseudonymized cohort, source-to-aria, and split manifests.
- Out-of-fold predictions for all methods in the main result table.
- MERT/MuQ layer selections and class-bias selections.
- Inner-development class-support diagnostics.
- Source/aria weighting, rare-class, bootstrap, and permutation summaries.
- Training and analysis scripts used by the locked experiment.

## What Is Not Included

- SVQTD audio, YouTube URLs, source identifiers, or singer identities.
- Third-party foundation-model weights.
- Cached MERT, MuQ, BEATs, or openSMILE feature matrices.
- Cloud credentials, SSH helpers, checkpoints, and private absolute paths.
- The authors obtained permission from the SVQTD provider to redistribute the
pseudonymized segment-level labels and derived prediction records included in
this repository.

Obtain SVQTD through the original dataset publication and follow its terms.
The pseudonymous ID algorithm is documented so authorized users can recreate
the mapping from their local metadata.

## Quick Reproduction

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_release.py
python scripts/reproduce_metrics.py
python scripts/summarize_inner_support.py
```

Generated files are written to `reproduced/`. The lightweight path uses the
archived out-of-fold predictions and does not require a GPU or source audio.

## Full Training

The scripts in `src/paper_pipeline/` document the complete analysis stages.
Full retraining additionally requires:

1. legitimately obtained SVQTD audio and metadata;
2. locally extracted foundation-model embeddings;
3. PyTorch and the model-specific dependencies in
   `requirements-training.txt`;
4. paths supplied through command-line arguments.

No outer-test labels are used for layer choice, early stopping, class-bias
selection, or calibration.

## Metric Rule

Inner model selection uses classes present in the inner-development
partition. All 30 generated inner-development partitions contain every
globally defined class, although the rarest open-throat class has as few as
two development examples. Primary pooled out-of-fold reporting uses the
globally predefined class set.

## Data and Licensing

Code is released under the MIT License. The license does not grant rights to
SVQTD audio, original annotations, foundation-model weights, or other
third-party materials. Read `docs/DATA_LICENSE_NOTICE.md` before publication
or reuse.

## Citation

Citation metadata are provided in `CITATION.cff`. Replace the repository and
article placeholders after the final GitHub release and Zenodo deposit.

