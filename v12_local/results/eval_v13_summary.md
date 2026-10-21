# eval_v13: summary

## test_full (879 images)

- s@1024+TTA vs baseline, mAP50_95: +0.029, 95% CI [+0.012, +0.046], p = 0.013, Holm p = 0.058 -> significant, but not after Holm
- s@1024+TTA vs baseline, B: +0.018, 95% CI [-0.039, +0.075], p = 0.315, Holm p = 0.315 -> NOT significant
- s@1024+TTA vs baseline, ALL: +0.017, 95% CI [+0.006, +0.028], p = 0.012, Holm p = 0.058 -> significant, but not after Holm
- s@1024 vs baseline, mAP50_95: -0.034, 95% CI [-0.051, -0.017], p = 0.005, Holm p = 0.027 -> significant
- s@1024 vs baseline, B: +0.016, 95% CI [-0.041, +0.073], p = 0.365, Holm p = 0.365 -> NOT significant

## Density law (baseline, per-class mAP@50 vs boxes per image)

- valid_full: r = -0.879 (p = 1.5e-07), 21 classes
- test_full: r = -0.896 (p = 4.8e-07), 18 classes
