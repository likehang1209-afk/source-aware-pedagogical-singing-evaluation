# Data and Third-Party Materials Notice

This repository does not redistribute SVQTD audio, source URLs, original
recording identifiers, singer identities, foundation-model weights, or cached
third-party embeddings.

The pseudonymized manifests and out-of-fold prediction files contain derived
research metadata needed to reproduce the reported statistical analyses.
Before making this release candidate public, the authors must confirm that the
SVQTD provider permits redistribution of segment-level annotation labels and
derived prediction records. If that permission is not explicit, remove:

- `data_manifests/cohort_pseudonymized.csv.gz`;
- `predictions/main_oof_predictions.csv.gz`;

and distribute the scripts plus aggregate outputs only. Authorized dataset
users can regenerate these files locally.

The MIT License applies only to original code in this repository. It does not
override the licenses or terms of SVQTD, MERT, MuQ, BEATs, openSMILE, PyTorch,
or any source platform.

