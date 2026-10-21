# twinfree_v12: summary

## test_full (1715 images)

- s@1024+TTA vs baseline, mAP50_95: +0.032, 95% CI [+0.021, +0.044], p = 0.005, Holm p = 0.027 -> significant
- s@1024+TTA vs baseline, B: +0.036, 95% CI [+0.002, +0.071], p = 0.044, Holm p = 0.174 -> significant, but not after Holm
- s@1024+TTA vs baseline, ALL: +0.011, 95% CI [-0.005, +0.027], p = 0.099, Holm p = 0.296 -> NOT significant
- s@1024 vs baseline, mAP50_95: -0.033, 95% CI [-0.043, -0.023], p = 0.001, Holm p = 0.007 -> significant
- s@1024 vs baseline, B: +0.040, 95% CI [+0.005, +0.074], p = 0.034, Holm p = 0.138 -> significant, but not after Holm

## test_twinfree (566 images)

- s@1024+TTA vs baseline, mAP50_95: +0.026, 95% CI [+0.014, +0.038], p = 0.004, Holm p = 0.019 -> significant
- s@1024+TTA vs baseline, B: +0.020, 95% CI [-0.014, +0.054], p = 0.161, Holm p = 0.482 -> NOT significant
- s@1024+TTA vs baseline, ALL: +0.005, 95% CI [-0.013, +0.022], p = 0.381, Holm p = 0.482 -> NOT significant
- s@1024 vs baseline, mAP50_95: -0.042, 95% CI [-0.055, -0.030], p = 0.003, Holm p = 0.017 -> significant
- s@1024 vs baseline, B: +0.023, 95% CI [-0.014, +0.059], p = 0.129, Holm p = 0.312 -> NOT significant

## test_twinned (1140 images)

- s@1024+TTA vs baseline, mAP50_95: +0.036, 95% CI [+0.030, +0.042], p = 0.000, Holm p = 0.000 -> significant
- s@1024+TTA vs baseline, B: +0.087, 95% CI [+0.080, +0.095], p = 0.000, Holm p = 0.000 -> significant
- s@1024+TTA vs baseline, ALL: +0.028, 95% CI [+0.026, +0.031], p = 0.000, Holm p = 0.000 -> significant
- s@1024 vs baseline, mAP50_95: -0.034, 95% CI [-0.039, -0.028], p = 0.000, Holm p = 0.000 -> significant
- s@1024 vs baseline, B: +0.086, 95% CI [+0.076, +0.097], p = 0.000, Holm p = 0.000 -> significant

## Density law (baseline, per-class mAP@50 vs boxes per image)

- valid_full: r = -0.921 (p = 1.2e-09), 22 classes
- valid_twinfree: r = -0.872 (p = 1.1e-06), 19 classes
- test_full: r = -0.851 (p = 1.0e-06), 21 classes
- test_twinfree: r = -0.734 (p = 3.4e-04), 19 classes

## Leakage effect (test, same class, twinned - twin-free AP@50)

- baseline: +0.004
- s@1024: +0.039
- s@1024+TTA: +0.036
