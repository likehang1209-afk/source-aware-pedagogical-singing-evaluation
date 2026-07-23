# Reporting Strengthening Summary

No model was retrained. All values were recomputed from locked out-of-fold predictions.

## Alternative metric estimands

| scheme          | estimand              |    uar |   macro_f1 |   accuracy |
|:----------------|:----------------------|-------:|-----------:|-----------:|
| MuQ             | segment_weighted      | 0.3596 |     0.3578 |     0.6221 |
| MuQ             | source_equal_weighted | 0.3572 |     0.3543 |     0.6277 |
| MuQ             | aria_equal_weighted   | 0.3572 |     0.3545 |     0.6186 |
| Adjusted MuQ    | segment_weighted      | 0.3781 |     0.3580 |     0.5476 |
| Adjusted MuQ    | source_equal_weighted | 0.3770 |     0.3543 |     0.5516 |
| Adjusted MuQ    | aria_equal_weighted   | 0.3744 |     0.3554 |     0.5471 |
| Adjusted fusion | segment_weighted      | 0.3841 |     0.3574 |     0.5355 |
| Adjusted fusion | source_equal_weighted | 0.3838 |     0.3546 |     0.5404 |
| Adjusted fusion | aria_equal_weighted   | 0.3870 |     0.3603 |     0.5425 |

## Rare-class ranking diagnostics

| scheme          | target          |   positive_class |   prevalence |   precision |   recall |   false_positive_rate |   average_precision |   no_information_ap |
|:----------------|:----------------|-----------------:|-------------:|------------:|---------:|----------------------:|--------------------:|--------------------:|
| MuQ             | chest_resonance |                0 |       0.0324 |      0.0800 |   0.0357 |                0.0138 |              0.0566 |              0.0324 |
| MuQ             | head_resonance  |                0 |       0.0420 |      0.0500 |   0.0069 |                0.0057 |              0.0542 |              0.0420 |
| MuQ             | front_placement |                2 |       0.0775 |      0.0964 |   0.0597 |                0.0471 |              0.0780 |              0.0775 |
| MuQ             | back_placement  |                2 |       0.0370 |      0.1190 |   0.0391 |                0.0111 |              0.0528 |              0.0370 |
| MuQ             | open_throat     |                3 |       0.0156 |      0.0588 |   0.0185 |                0.0047 |              0.0314 |              0.0156 |
| MuQ             | breathiness     |                1 |       0.1143 |      0.1323 |   0.1089 |                0.0921 |              0.1211 |              0.1143 |
| MuQ             | vibrato         |                2 |       0.0333 |      0.0278 |   0.0261 |                0.0314 |              0.0538 |              0.0333 |
| Adjusted MuQ    | chest_resonance |                0 |       0.0324 |      0.0629 |   0.1875 |                0.0936 |              0.0521 |              0.0324 |
| Adjusted MuQ    | head_resonance  |                0 |       0.0420 |      0.0522 |   0.0828 |                0.0658 |              0.0521 |              0.0420 |
| Adjusted MuQ    | front_placement |                2 |       0.0775 |      0.1053 |   0.2164 |                0.1546 |              0.0974 |              0.0775 |
| Adjusted MuQ    | back_placement  |                2 |       0.0370 |      0.0489 |   0.1406 |                0.1052 |              0.0517 |              0.0370 |
| Adjusted MuQ    | open_throat     |                3 |       0.0156 |      0.0156 |   0.0185 |                0.0185 |              0.0267 |              0.0156 |
| Adjusted MuQ    | breathiness     |                1 |       0.1143 |      0.1108 |   0.2354 |                0.2437 |              0.1120 |              0.1143 |
| Adjusted MuQ    | vibrato         |                2 |       0.0333 |      0.0870 |   0.2087 |                0.0754 |              0.0573 |              0.0333 |
| Adjusted fusion | chest_resonance |                0 |       0.0324 |      0.0649 |   0.2232 |                0.1077 |              0.0638 |              0.0324 |
| Adjusted fusion | head_resonance  |                0 |       0.0420 |      0.0714 |   0.0552 |                0.0314 |              0.0569 |              0.0420 |
| Adjusted fusion | front_placement |                2 |       0.0775 |      0.1141 |   0.2201 |                0.1437 |              0.0935 |              0.0775 |
| Adjusted fusion | back_placement  |                2 |       0.0370 |      0.0654 |   0.3047 |                0.1674 |              0.0675 |              0.0370 |
| Adjusted fusion | open_throat     |                3 |       0.0156 |      0.0165 |   0.0741 |                0.0700 |              0.0175 |              0.0156 |
| Adjusted fusion | breathiness     |                1 |       0.1143 |      0.1210 |   0.3468 |                0.3251 |              0.1186 |              0.1143 |
| Adjusted fusion | vibrato         |                2 |       0.0333 |      0.0360 |   0.0783 |                0.0721 |              0.0386 |              0.0333 |

## Selection stability

| target          |   mean_mode_agreement |   unanimous_folds |   mean_unique_tau_count |
|:----------------|----------------------:|------------------:|------------------------:|
| back_placement  |                 0.500 |                 0 |                   3.333 |
| breathiness     |                 0.467 |                 0 |                   3.167 |
| chest_resonance |                 0.467 |                 0 |                   2.833 |
| front_placement |                 0.533 |                 0 |                   3.000 |
| head_resonance  |                 0.600 |                 0 |                   3.000 |
| open_throat     |                 0.533 |                 0 |                   3.167 |
| vibrato         |                 0.400 |                 0 |                   3.833 |

## Permutation validity

| role   |   fold_permutations |   movable_fraction_min |   movable_fraction_max |   singleton_rows_total |   expected_fixed_fraction_mean |
|:-------|--------------------:|-----------------------:|-----------------------:|-----------------------:|-------------------------------:|
| dev    |                 120 |                 1.0000 |                 1.0000 |                      0 |                         0.0671 |
| test   |                 120 |                 1.0000 |                 1.0000 |                      0 |                         0.0735 |
| train  |                 120 |                 1.0000 |                 1.0000 |                      0 |                         0.0659 |