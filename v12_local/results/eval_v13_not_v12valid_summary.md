# eval_v13_not_v12valid: summary

## test_full (879 images)

- s@1024+TTA vs baseline, mAP50_95: +0.029, 95% CI [+0.012, +0.046], p = 0.013, Holm p = 0.058 -> significant, but not after Holm
- s@1024+TTA vs baseline, B: +0.021, 95% CI [-0.040, +0.081], p = 0.308, Holm p = 0.308 -> NOT significant
- s@1024+TTA vs baseline, ALL: +0.017, 95% CI [+0.006, +0.028], p = 0.012, Holm p = 0.058 -> significant, but not after Holm
- s@1024 vs baseline, mAP50_95: -0.034, 95% CI [-0.051, -0.017], p = 0.005, Holm p = 0.027 -> significant
- s@1024 vs baseline, B: +0.019, 95% CI [-0.043, +0.082], p = 0.329, Holm p = 0.329 -> NOT significant

## test_not_v12valid (699 images)

- s@1024+TTA vs baseline, mAP50_95: +0.025, 95% CI [+0.012, +0.039], p = 0.007, Holm p = 0.036 -> significant
- s@1024+TTA vs baseline, B: +0.015, 95% CI [-0.005, +0.035], p = 0.092, Holm p = 0.184 -> NOT significant
- s@1024+TTA vs baseline, ALL: +0.015, 95% CI [+0.001, +0.030], p = 0.043, Holm p = 0.130 -> significant, but not after Holm
- s@1024 vs baseline, mAP50_95: -0.037, 95% CI [-0.052, -0.022], p = 0.003, Holm p = 0.013 -> significant
- s@1024 vs baseline, B: +0.011, 95% CI [-0.007, +0.030], p = 0.145, Holm p = 0.394 -> NOT significant

## Density law (baseline, per-class mAP@50 vs boxes per image)

- test_full: r = -0.896 (p = 4.8e-07), 18 classes
