# Grid Report v2: Variability Decomposition Correction (supersedes the "Variability" section of GRID_REPORT.md)

Date: 2026-10-10. `GRID_REPORT.md` is preserved unchanged. No model, prediction or metric file was altered; this is a re-summary of
`grid_v1/cell_metrics_by_repeat_draw.csv`. New tables: `grid_v1/variability_decomposition_v2.csv`, `grid_v1/contrast_variability_decomposition_v2.csv`.

## Correction
The column `between_draw_sd_pooled` in `learning_surface.csv` is the SD over all 60 (repeat x draw) values and therefore mixes draw-to-draw variability within a
repeat with repeat-to-repeat variability of the outer split. It should not be read as acquisition-selection variability alone. The decomposition below separates them.

Definitions: for each cell, the 20 draw values within repeat r give `within_repeat_sd_r` (SD over draws, ddof=1); `within_repeat_sd_mean` is the mean of the three;
`sd_of_repeat_means` is the SD (ddof=1) of the three draw-averaged repeat means. The same is done for the paired difference series.

## Cells (balanced accuracy)
| n | k | within-repeat SD (mean of 3) | SD of repeat means | previous pooled column |
|---:|---:|---:|---:|---:|
| 21 | 1 | 0.047 | 0.013 | 0.047 |
| 21 | 3 | 0.047 | 0.008 | 0.048 |
| 21 | 5 | 0.044 | 0.012 | 0.045 |
| 28 | 1 | 0.043 | 0.009 | 0.043 |
| 28 | 3 | 0.037 | 0.013 | 0.039 |
| 28 | 5 | 0.041 | 0.016 | 0.042 |
| 35 | 1 | 0.040 | 0.016 | 0.042 |
| 35 | 3 | 0.035 | 0.018 | 0.038 |
| 35 | 5 | 0.035 | 0.013 | 0.037 |
| 42 | 1 | 0.034 | 0.015 | 0.036 |
| 42 | 3 | 0.029 | 0.012 | 0.030 |
| 42 | 5 | 0.033 | 0.009 | 0.034 |
| 49 | 1 | 0.037 | 0.014 | 0.038 |
| 49 | 3 | 0.033 | 0.014 | 0.034 |
| 49 | 5 | 0.032 | 0.014 | 0.034 |
| 56 | 1 | 0.024 | 0.013 | 0.027 |
| 56 | 3 | 0.021 | 0.015 | 0.024 |
| 56 | 5 | 0.000 | 0.019 | 0.016 |

Reading: within-repeat (acquisition-draw) SD falls from about 0.047 at n=21 to about 0.021 at n=56 with k=3, and is exactly 0 at 56x5 because only one composition exists;
the earlier pooled value of 0.016 for 56x5 was entirely repeat-to-repeat variability. SD of repeat means stays around 0.01 to 0.02 across cells.
Three repeats give a very imprecise SD of repeat means; it is descriptive only.

## Prespecified contrasts (paired BA difference)
| Contrast | Mean | Within-repeat SD r0 / r1 / r2 | SD of repeat means | Repeat means r0 / r1 / r2 | Pairs > 0 per repeat |
|---|---:|---|---:|---|---|
| Primary 35x3 − 21x5 | +0.037 | 0.052 / 0.060 / 0.043 | 0.006 | +0.042 / +0.038 / +0.031 | 14/20, 15/20, 16/20 |
| Secondary 56x3 − 35x5 | +0.035 | 0.038 / 0.038 / 0.053 | 0.004 | +0.035 / +0.038 / +0.031 | 16/20, 17/20, 14/20 |

Reading: the paired difference varies mainly from draw to draw (SD about 0.05) and very little between repeats (SD about 0.005); the direction is the same in all
three repeats. The bootstrap CI in GRID_REPORT.md remains the participant-level uncertainty statement; these SDs describe acquisition-selection and split variability
on the same 70 people and are not independent-cohort uncertainty.
