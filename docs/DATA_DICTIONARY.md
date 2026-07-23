# Data Dictionary

## Pseudonymous identifiers

`segment_uid` and `source_uid` are the first 24 hexadecimal characters of:

```text
SHA256("svqtd-source-aware-v1|segment|<original segment ID>")
SHA256("svqtd-source-aware-v1|source|<original source ID>")
```

They are linkage pseudonyms, not claims of irreversible anonymization.

## Cohort manifest

`data_manifests/cohort_pseudonymized.csv.gz` contains one row per retained
segment: pseudonymous IDs, aria, outer fold, original released split label,
and seven categorical targets.

The dataset column called `breathiness` is retained for file compatibility.
The article refers to it as Dataset B/R because the dataset author confirmed
that the CSV field corresponds to the paper's roughness label and was not
annotated using a strict breathiness-versus-roughness distinction.

## Inner roles

`data_manifests/repeated_inner_roles.csv.gz` contains the six outer folds and
five source-disjoint inner repetitions. `outer_test` is fixed for each outer
fold; `inner_train` and `inner_dev` vary by repetition.

## Predictions

`predictions/main_oof_predictions.csv.gz` contains:

- `method`: manuscript method label;
- `outer_fold`, `aria`: held-repertoire assignment;
- `segment_uid`, `source_uid`: pseudonymous linkage keys;
- `target`: one of seven attributes;
- `y_true`, `y_pred`: integer class labels;
- `probability`: JSON array of class scores.

Each method contains exactly 3,456 x 7 = 24,192 rows.

