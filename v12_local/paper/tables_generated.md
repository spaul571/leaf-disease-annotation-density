# Tables (merged-dataset-v13)

## Table 1

Per-class composition and annotation statistics of the benchmark, with baseline accuracy for
reference. Protocol A is the whole-leaf convention and B the lesion convention: a class is in B
when its median box covers less than half the image. `Images` counts images containing at least
one box of that class, so the column sums to more than the 8,785 images in the benchmark
because a few images carry boxes of two classes. `Box area` is the median annotated box area as
a fraction of image area. `Mixed` is the fraction of a class's boxes drawn in the opposite
convention from its own majority, and is a lower bound (Section 3.3). `mAP@50` is the mean of
three baseline replicates on the validation split. Rows are ordered by protocol group, then by
descending annotation density.

| Class | Protocol | Images | Boxes | Boxes per image | Box area | Mixed | mAP@50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cucumber Downy Mildew | A (whole-leaf) | 544 | 742 | 1.36 | 0.984 | 0.228 | 0.762 |
| cauliflower black rot | A (whole-leaf) | 51 | 64 | 1.25 | 0.682 | 0.078 | 0.837 |
| potato healthy | A (whole-leaf) | 459 | 521 | 1.14 | 0.672 | 0.092 | 0.963 |
| cauliflower downy mildew | A (whole-leaf) | 158 | 171 | 1.08 | 0.882 | 0.012 | 0.934 |
| cauliflower no disease | A (whole-leaf) | 183 | 192 | 1.05 | 1.000 | 0.042 | 0.991 |
| bean healthy | A (whole-leaf) | 44 | 45 | 1.02 | 0.693 | 0.000 | 0.995 |
| tomato curl virus | A (whole-leaf) | 1500 | 1514 | 1.01 | 0.839 | 0.000 | 0.995 |
| cauliflower bacterial spot rot | A (whole-leaf) | 162 | 162 | 1.00 | 0.871 | 0.000 | 0.970 |
| chilli Curling-Leafs | A (whole-leaf) | 33 | 33 | 1.00 | 0.706 | 0.000 | 0.937 |
| chilli Healthy-Leafs | A (whole-leaf) | 63 | 63 | 1.00 | 0.934 | 0.000 | 0.995 |
| chilli Leaf Spot-Leafs | A (whole-leaf) | 22 | 22 | 1.00 | 0.809 | 0.000 | 0.829 |
| chilli Leveillula taurica-Leafs | A (whole-leaf) | 29 | 29 | 1.00 | 0.897 | 0.000 | 0.980 |
| chilli Mozaik Leaf | A (whole-leaf) | 74 | 74 | 1.00 | 0.917 | 0.000 | 0.995 |
| tomato bacterial spot | A (whole-leaf) | 1497 | 1499 | 1.00 | 0.923 | 0.000 | 0.994 |
| tomato two spotted spider mite | A (whole-leaf) | 1497 | 1497 | 1.00 | 0.756 | 0.000 | 0.994 |
| potato blackleg | B (lesion) | 139 | 978 | 7.04 | 0.008 | 0.000 | 0.317 |
| cucumber Bacterial Wilt | B (lesion) | 732 | 3550 | 4.85 | 0.135 | 0.120 | 0.726 |
| potato early blight | B (lesion) | 258 | 1114 | 4.32 | 0.013 | 0.025 | 0.666 |
| cucumber Antraknosa | B (lesion) | 789 | 1771 | 2.24 | 0.043 | 0.404 | 0.705 |
| potato late blight | B (lesion) | 495 | 936 | 1.89 | 0.030 | 0.317 | 0.647 |
| bean angular leaf spot | B (lesion) | 43 | 79 | 1.84 | 0.260 | 0.127 | 0.875 |
| bean bean rust | B (lesion) | 46 | 78 | 1.70 | 0.215 | 0.064 | 0.925 |


## Table 2

The measured noise floor. Standard deviations are computed across three baseline replicates
differing only in random seed. The 95% bar for a single run against the three-seed mean is
t(0.975, 2)·sd·√(1 + 1/3), and for two single runs t(0.975, 2)·sd·√2, with t(0.975, 2) = 4.303.
Comparisons of means at three seeds per arm are made with Welch's t-test instead (Table 5). The
lesion group (B) is far noisier on the test split than on validation because the test split
contains few lesion-class instances (Section 4.2).

| Metric | Validation sd | Validation bar, single run vs mean | Validation bar, two single runs | Test sd | Test bar, two single runs |
| --- | --- | --- | --- | --- | --- |
| Class mean mAP@50 | 0.0068 | 0.0336 | 0.0411 | 0.0048 | 0.0291 |
| Group A mAP@50 | 0.0065 | 0.0324 | 0.0397 | 0.0095 | 0.0577 |
| Group B mAP@50 | 0.0080 | 0.0396 | 0.0485 | 0.0264 | 0.1605 |
| Recall | 0.0138 | 0.0687 | 0.0841 | 0.0085 | 0.0520 |
| mAP@50–95 | 0.0041 | 0.0203 | 0.0249 | 0.0081 | 0.0494 |


## Table 3

Every arm on the v13 validation split. `baseline` is the three-seed mean; every other arm is
compared with it (Table 4).

| Arm | Seeds | Class mean mAP@50 | Group A | Group B | Recall | mAP@50–95 | A−B gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 3 | 0.8650 | 0.9446 | 0.6945 | 0.8333 | 0.6928 | 0.2501 |
| s @640 | 1 | 0.8665 | 0.9428 | 0.7032 | 0.8392 | 0.6999 | 0.2396 |
| m @640 | 1 | 0.8735 | 0.9497 | 0.7101 | 0.8551 | 0.7095 | 0.2397 |
| RT-DETR-l | 1 | 0.8737 | 0.9471 | 0.7165 | 0.8389 | 0.6935 | 0.2306 |
| +WIoU only | 1 | 0.8675 | 0.9557 | 0.6785 | 0.8332 | 0.6929 | 0.2773 |
| +P2 head | 1 | 0.8704 | 0.9478 | 0.7048 | 0.8233 | 0.6918 | 0.2430 |
| +P2+WIoU | 1 | 0.8755 | 0.9622 | 0.6896 | 0.8332 | 0.6963 | 0.2726 |
| +UADH | 1 | 0.8540 | 0.9464 | 0.6558 | 0.8027 | 0.6888 | 0.2906 |
| oversample | 1 | 0.8559 | 0.9397 | 0.6765 | 0.8293 | 0.6803 | 0.2632 |
| naive over+under | 1 | 0.8660 | 0.9515 | 0.6828 | 0.8349 | 0.6891 | 0.2686 |
| focal loss | 1 | 0.8450 | 0.9337 | 0.6550 | 0.8043 | 0.6769 | 0.2788 |
| n @1024 | 1 | 0.8741 | 0.9501 | 0.7113 | 0.8377 | 0.6491 | 0.2389 |
| s @1024 | 3 | 0.8745 | 0.9547 | 0.7028 | 0.8549 | 0.6561 | 0.2519 |


## Table 4

Every effect against the three-seed baseline, on the validation split. Values are standardised
effects: the change in that metric divided by its own 95% threshold, carrying the sign of the
change. The threshold is the exact bar of Table 2 for a single-seed arm and the half-width of
Welch's interval for a multi-seed arm. A magnitude of 1 or more lies beyond seed noise and is
shown in bold. Absolute values for every arm are given in Table 3.

| Arm | Class mean mAP@50 | Group A | Group B | Recall | mAP@50–95 |
| --- | --- | --- | --- | --- | --- |
| +P2 head | +0.2 | +0.1 | +0.3 | -0.1 | -0.0 |
| +P2+WIoU | +0.3 | +0.5 | -0.1 | -0.0 | +0.2 |
| +UADH | -0.3 | +0.1 | -1.0 | -0.4 | -0.2 |
| +WIoU only | +0.1 | +0.3 | -0.4 | -0.0 | +0.0 |
| RT-DETR-l | +0.3 | +0.1 | +0.6 | +0.1 | +0.0 |
| focal loss | -0.6 | -0.3 | -1.0 | -0.4 | -0.8 |
| m @640 | +0.3 | +0.2 | +0.4 | +0.3 | +0.8 |
| n @1024 | +0.3 | +0.2 | +0.4 | +0.1 | **-2.2** |
| naive over+under | +0.0 | +0.2 | -0.3 | +0.0 | -0.2 |
| oversample | -0.3 | -0.2 | -0.5 | -0.1 | -0.6 |
| s @1024 | +0.7 | +0.7 | +0.5 | +0.7 | **-4.3** |
| s @640 | +0.0 | -0.1 | +0.2 | +0.1 | +0.3 |


## Table 5

Held-out test split, the result of record. Three seeds per arm, reported as mean ± standard
deviation, with Welch's two-sample *t*-test (unequal variances) against the baseline for every
metric, and the Holm-adjusted p-value across the five metrics of each comparison. The
configuration was selected on the validation split (Section 3.6, Section 4.5) and measured once
here.

| System | Class mean mAP@50 | Group A | Group B | Recall | mAP@50–95 |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.876 ± 0.005 | 0.952 ± 0.009 | 0.715 ± 0.026 | 0.822 ± 0.009 | 0.704 ± 0.008 |
| YOLOv11s 1024 | 0.892 ± 0.003 | 0.966 ± 0.006 | 0.735 ± 0.005 | 0.844 ± 0.014 | 0.670 ± 0.006 |
| YOLOv11s 1024 + TTA | 0.893 ± 0.005 | 0.967 ± 0.010 | 0.736 ± 0.007 | 0.856 ± 0.003 | 0.733 ± 0.004 |
| YOLOv11s 1024, Welch p | 0.013 | 0.110 | 0.329 | 0.102 | 0.005 |
| YOLOv11s 1024 + TTA, Welch p | 0.012 | 0.128 | 0.308 | 0.012 | 0.013 |
| YOLOv11s 1024, Holm p | 0.053 | 0.306 | 0.329 | 0.306 | 0.027 |
| YOLOv11s 1024 + TTA, Holm p | 0.058 | 0.256 | 0.308 | 0.058 | 0.058 |


## Table 6

Does annotation density survive controls for object size, class frequency and source dataset?
All rows use per-class mAP@50 averaged over the three baseline replicates on the validation
split, n = 22 classes. The multiple regression attains R² = 0.776 (adjusted 0.739). Each of the
six source datasets contributed exactly one crop, so crop and source dataset are the same
factor in this benchmark and cannot be separated; conditioning on crop is therefore the
available source control. A crop variance estimated at zero is a singular fit, meaning crop
explains no residual variance once density is in the model. These are controls on an
observational design, not evidence of causation (Section 4.1).

| Analysis | Estimate | p | Note |
| --- | --- | --- | --- |
| Univariate: boxes per image (density) | r = -0.867 | 1.8 × 10⁻⁷ | Spearman -0.782 |
| Univariate: median box area | r = +0.746 | 6.8 × 10⁻⁵ | Spearman +0.685 |
| Univariate: class frequency (log boxes) | r = -0.414 | 0.056 | Spearman -0.301 |
| Univariate: within-class protocol mixing | r = -0.401 | 0.064 | Spearman -0.614 |
| Partial correlation, density vs mAP@50, controlling median box area | r = -0.704 | 3.6 × 10⁻⁴ | df = 19 |
| Partial correlation, density vs mAP@50, controlling class frequency | r = -0.836 | 2.3 × 10⁻⁶ | df = 19 |
| Partial correlation, density vs mAP@50, controlling both | r = -0.692 | 7.3 × 10⁻⁴ | df = 18 |
| Multiple regression: density | β = -0.076 ± 0.019 | 7.3 × 10⁻⁴ | VIF 2.38 |
| Multiple regression: median box area | β = +0.110 ± 0.077 | 0.170 | VIF 2.20 |
| Multiple regression: class frequency | β = +0.003 ± 0.031 | 0.919 | VIF 1.30 |
| Density with crop (= source dataset) as fixed effect | β = -0.071 ± 0.016 | 4.9 × 10⁻⁴ | R² = 0.821 |
| Density with crop as random intercept | β = -0.093 | n/a | did not converge; standard error not estimable |


## Table 7

Individual seed results behind every multi-seed mean reported in this paper. Arms differ only
in the random seed passed to the training framework. The validation rows are the three baseline
replicates that define the noise floor of Table 2; the test rows are the three arms of Table 5.
At three seeds per arm a single-arm 95% interval uses t(2) = 4.303, so intervals are reported
on differences between arms rather than on individual means (Section 3.8).

| Split | Arm | Seed | Class mean mAP@50 | Group A | Group B | Recall | mAP@50–95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| validation | baseline (640 px) | 0 | 0.8613 | 0.9425 | 0.6875 | 0.8296 | 0.6881 |
| validation | baseline (640 px) | 1 | 0.8609 | 0.9394 | 0.6928 | 0.8217 | 0.6948 |
| validation | baseline (640 px) | 2 | 0.8728 | 0.9519 | 0.7032 | 0.8486 | 0.6955 |
| test | baseline (640 px) | 0 | 0.8715 | 0.9574 | 0.6874 | 0.8257 | 0.6978 |
| test | baseline (640 px) | 1 | 0.8811 | 0.9569 | 0.7185 | 0.8117 | 0.7009 |
| test | baseline (640 px) | 2 | 0.8768 | 0.9408 | 0.7398 | 0.8273 | 0.7132 |
| test | YOLOv11s 1024 | 0 | 0.8912 | 0.9614 | 0.7408 | 0.8336 | 0.6693 |
| test | YOLOv11s 1024 | 1 | 0.8899 | 0.9632 | 0.7329 | 0.8602 | 0.6642 |
| test | YOLOv11s 1024 | 2 | 0.8954 | 0.9724 | 0.7304 | 0.8368 | 0.6760 |
| test | YOLOv11s 1024 + TTA | 0 | 0.8879 | 0.9554 | 0.7434 | 0.8597 | 0.7295 |
| test | YOLOv11s 1024 + TTA | 1 | 0.8964 | 0.9721 | 0.7342 | 0.8538 | 0.7329 |
| test | YOLOv11s 1024 + TTA | 2 | 0.8962 | 0.9738 | 0.7298 | 0.8543 | 0.7368 |


## Table 8

Dataloader worker-count control, run on v12 before the v13 rebuild; every v13 run uses two
workers. The 640 px baseline was retrained with two worker processes instead of eight, three
seeds, every other setting held fixed, and compared against the three-seed eight-worker
baseline. The bar is 1.96·sd_pooled·√(2/3), the same mean-versus-mean rule used elsewhere in
this paper, and the ratio is the difference divided by that bar. No metric reaches its bar, so
worker count carries no effect detectable at three seeds per arm. The two sets of runs were
measured two months apart on the same machine, so this null also covers environment drift over
that interval. Mean wall-clock per 100-epoch run on the same machine: YOLOv11s 1024 (workers=2,
selected), 4.92 h; baseline 640 (workers=2, control), 1.95 h; baseline 640 (workers=8,
published), 2.34 h.

| Metric | 8 workers | 2 workers | Difference | 95% bar | Ratio | Welch p |
| --- | --- | --- | --- | --- | --- | --- |
| Class mean mAP@50 | 0.8990 ± 0.0015 | 0.8982 ± 0.0038 | -0.0007 | 0.0081 | 0.09 | 0.777 |
| Group A | 0.9677 ± 0.0017 | 0.9661 ± 0.0028 | -0.0017 | 0.0056 | 0.29 | 0.436 |
| Group B | 0.7516 ± 0.0031 | 0.7528 ± 0.0099 | +0.0012 | 0.0222 | 0.06 | 0.854 |
| Recall | 0.8576 ± 0.0021 | 0.8623 ± 0.0174 | +0.0047 | 0.0424 | 0.11 | 0.685 |
| mAP@50–95 | 0.7355 ± 0.0052 | 0.7292 ± 0.0045 | -0.0063 | 0.0111 | 0.56 | 0.192 |
