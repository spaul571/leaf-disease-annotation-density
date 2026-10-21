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
differing only in random seed. The significance bar for a difference between two single runs is
1.96·sd·√2. Comparisons of means at three seeds per arm on the test split are made with Welch's
t-test instead (Table 5), because the pooled standard deviation differs for each pair of arms
and no single bar applies. The lesion group (B) is far noisier on the test split than on
validation because the test split contains few lesion-class instances (Section 4.2). The last
column is the v12 validation bar, against which the Environment 1 arms, run on v12 only, are
judged (Tables 3b and 4b).

| Metric | Validation sd | Validation bar | Test sd | Test bar (single run) | v12 validation bar |
| --- | --- | --- | --- | --- | --- |
| Class mean mAP@50 | 0.0068 | 0.0187 | 0.0048 | 0.0133 | 0.0040 |
| Group A mAP@50 | 0.0065 | 0.0181 | 0.0095 | 0.0263 | 0.0047 |
| Group B mAP@50 | 0.0080 | 0.0221 | 0.0264 | 0.0731 | 0.0085 |
| Recall | 0.0138 | 0.0383 | 0.0085 | 0.0237 | 0.0057 |
| mAP@50–95 | 0.0041 | 0.0113 | 0.0081 | 0.0225 | 0.0145 |


## Table 3

All Environment 2 arms on the v13 validation split. `baseline` is the three-seed mean; every
other arm is compared with it (Table 4). The Environment 1 arms were run on v12 only and are
reported separately (Table 3b).

| Arm | Seeds | Class mean mAP@50 | Group A | Group B | Recall | mAP@50–95 | A−B gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 3 | 0.8650 | 0.9446 | 0.6945 | 0.8333 | 0.6928 | 0.2501 |
| +P2 head | 1 | 0.7918 | 0.9172 | 0.5232 | 0.7681 | 0.6107 | 0.3940 |
| +P2+WIoU | 1 | 0.7849 | 0.9149 | 0.5064 | 0.7374 | 0.6068 | 0.4085 |
| +UADH | 1 | 0.8540 | 0.9464 | 0.6558 | 0.8027 | 0.6888 | 0.2906 |
| n @1024 | 1 | 0.8741 | 0.9501 | 0.7113 | 0.8377 | 0.6491 | 0.2389 |
| s @1024 | 3 | 0.8745 | 0.9547 | 0.7028 | 0.8549 | 0.6561 | 0.2519 |


## Table 4

Every Environment 2 effect against the three-seed baseline, on the validation split. Values are
standardised effects: the change in that metric divided by its own significance bar from Table
2, carrying the sign of the change. A magnitude of 2 or more is a real effect and is shown in
bold, 1 to 2 is marginal, and below 1 is indistinguishable from seed noise. Absolute values for
every arm are given in Table 3.

| Arm | Class mean mAP@50 | Group A | Group B | Recall | mAP@50–95 |
| --- | --- | --- | --- | --- | --- |
| +P2 head | **-3.9** | -1.5 | **-7.8** | -1.7 | **-7.2** |
| +P2+WIoU | **-4.3** | -1.6 | **-8.5** | **-2.5** | **-7.6** |
| +UADH | -0.6 | +0.1 | -1.8 | -0.8 | -0.3 |
| n @1024 | +0.5 | +0.3 | +0.8 | +0.1 | **-3.9** |
| s @1024 | +0.5 | +0.6 | +0.4 | +0.6 | **-3.2** |


## Table 3b

Environment 1 arms, run on v12 only. Their validation split shares rotated and flipped twins
with training (Section 3.2), so these results carry that caveat. `baseline (Environment 1)` is
the single-seed reference for the 100-epoch arms and `n @60ep` for the 60-epoch size sweep.

| Arm | Seeds | Class mean mAP@50 | Group A | Group B | Recall | mAP@50–95 | A−B gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (Environment 1) | 1 | 0.8975 | 0.9680 | 0.7465 | 0.8697 | 0.7268 | 0.2215 |
| oversample | 1 | 0.9023 | 0.9762 | 0.7439 | 0.8652 | 0.7398 | 0.2322 |
| naive over+under | 1 | 0.8837 | 0.9645 | 0.7104 | 0.8496 | 0.7191 | 0.2541 |
| focal loss | 1 | 0.8651 | 0.9487 | 0.6861 | 0.8088 | 0.7059 | 0.2626 |
| +WIoU only | 1 | 0.8982 | 0.9664 | 0.7521 | 0.8527 | 0.7322 | 0.2142 |
| n @60ep | 1 | 0.8809 | 0.9552 | 0.7216 | 0.8521 | 0.7083 | 0.2336 |
| s @60ep | 1 | 0.9044 | 0.9715 | 0.7606 | 0.8620 | 0.7416 | 0.2109 |
| m @60ep | 1 | 0.9066 | 0.9746 | 0.7609 | 0.8607 | 0.7501 | 0.2137 |
| RT-DETR-l @30ep | 1 | 0.8891 | 0.9619 | 0.7332 | 0.8546 | 0.7303 | 0.2286 |


## Table 4b

Environment 1 effects against their matched references, judged against the v12 validation bar
(Table 2, last column). Values are standardised effects: the change in that metric divided by
its own significance bar from Table 2, carrying the sign of the change. A magnitude of 2 or
more is a real effect and is shown in bold, 1 to 2 is marginal, and below 1 is
indistinguishable from seed noise. Reference arms: (b) the single-seed baseline in Environment
1, (c) the 60-epoch nano arm. Arms with no matched reference at their own epoch budget carry no
row.

| Arm | Class mean mAP@50 | Group A | Group B | Recall | mAP@50–95 |
| --- | --- | --- | --- | --- | --- |
| +WIoU only (b) | +0.2 | -0.3 | +0.7 | **-3.0** | +0.4 |
| focal loss (b) | **-8.0** | **-4.1** | **-7.1** | **-10.6** | -1.4 |
| m @60ep (c) | **+6.4** | **+4.1** | **+4.6** | +1.5 | **+2.9** |
| naive over+under (b) | **-3.4** | -0.7 | **-4.2** | **-3.5** | -0.5 |
| oversample (b) | +1.2 | +1.7 | -0.3 | -0.8 | +0.9 |
| s @60ep (c) | **+5.8** | **+3.4** | **+4.6** | +1.7 | **+2.3** |


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
| Class mean mAP@50 | 0.8990 ± 0.0015 | 0.8982 ± 0.0038 | -0.0007 | 0.0046 | 0.16 | 0.777 |
| Group A | 0.9677 ± 0.0017 | 0.9661 ± 0.0028 | -0.0017 | 0.0037 | 0.45 | 0.436 |
| Group B | 0.7516 ± 0.0031 | 0.7528 ± 0.0099 | +0.0012 | 0.0117 | 0.10 | 0.854 |
| Recall | 0.8576 ± 0.0021 | 0.8623 ± 0.0174 | +0.0047 | 0.0198 | 0.24 | 0.685 |
| mAP@50–95 | 0.7355 ± 0.0052 | 0.7292 ± 0.0045 | -0.0063 | 0.0078 | 0.80 | 0.192 |
