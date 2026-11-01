# Annotation density predicts per-class accuracy in leaf-disease detection

Code, split manifest, training configurations and result tables for the paper *Annotation density
predicts per-class accuracy in leaf-disease detection: a label-computable predictor and a measured
noise floor* (Shrikanta Paul and Shahadat Hossain, Daffodil International University). The
manuscript is under review.

Archived on Zenodo: [10.5281/zenodo.23539091](https://doi.org/10.5281/zenodo.23539091) (all
versions, resolves to the latest); version v1.1.0 is
[10.5281/zenodo.23858444](https://doi.org/10.5281/zenodo.23858444) and version v1.0.0 is
[10.5281/zenodo.23539092](https://doi.org/10.5281/zenodo.23539092).

## Changes in v1.1.0 (2026-11-01)

A revision after a second pre-submission review:

- **Every intervention is now trained on v13.** The variants that existed only on the earlier v12
  split were retrained on v13 against the three-seed baseline: oversampling, naive resampling,
  focal loss, WIoU, YOLOv11s and YOLOv11m at 640 px, and RT-DETR-l.
- **The earlier P2 runs trained from random weights.** P2 and P2+WIoU were retrained with their
  pretrained weights mapped by layer role. Ultralytics' `train()` rebuilds a model built from a
  YAML file and drops weights copied into it beforehand, which is what happened to the earlier P2
  runs (`v13_p2_*`, `local_p2_*`). See *Training and evaluation* below.
- **Noise bars are exact t-based 95% bars** (`paper_data.bar`): t(0.975, n−1)·sd·√(1 + 1/n) for a
  single run against an n-seed mean, and Welch's interval for multi-seed variants.
- **New analyses:**
  - robustness of the density relationship (`results/v13_reviewer_stats.json`, key
    `S2_robustness`);
  - the held-out comparison on the 699 test images outside the v12 validation split
    (`results/eval_v13_not_v12valid_*`);
  - a training-split diagnostic of the P2 runs (`results/v13_trainsplit_*`).

The images are not included. They belong to six public Roboflow Universe datasets, whose licences
govern their use (see [Source datasets](#source-datasets)). Everything else needed to rebuild the
benchmark splits and every table in the paper is here.

## What is in this repository

| Path | Contents |
|---|---|
| `dataset_preparation/` | Notebooks that merged the six source datasets into one 22-class label space and made the first split (`merged-dataset-v10`). Kept as run, with the authors' local paths. |
| `build_v12.py` | Stage 1 of duplicate removal: 64-bit dHash, Hamming distance ≤ 3, union–find, keep the image with most labels; re-split 70/20/10, seed 42. `v12_out.txt` is its log. |
| `v12_local/audit_dihedral_twins.py` | Finds rotated and flipped copies: hashes each image in all 8 orientations. Run on v12 it found the copies that stage 2 removes; run on v13 it finds none. |
| `build_v13.py` | Stage 2: collapses every rotated/flipped group to one image, then re-splits 70/20/10 by main class, seed 42. This is the benchmark of record. |
| `v13_manifest.csv` | **Split manifest**: one row per image with its split, its split in v12, SHA-256, dHash, main class, box count, crop and twin-group size. |
| `v13_distribution.csv`, `v12_distribution.csv` | Boxes and images per class and split. |
| `v12_local/` | Everything run on the local workstation (Environment 2): training notebooks, configurations, evaluation, statistics and the scripts that build every table and figure. |
| `v12_local/nb/` | One notebook per training run (outputs stripped). `v13_*` notebooks are the record; `local_*` notebooks are the same runs on the earlier v12 split. |
| `v12_local/runs/<run>/` | `args.yaml` (the full training configuration, including the seed) and `results.csv` (the per-epoch training curve) of every run. |
| `v12_local/results/` | Per-class results of every run (`perclass_*.csv`), test-split evaluations, statistics and analysis outputs. |
| `v12_local/paper/` | The tables of the paper as generated (`tables_generated.md` and `table*.csv`). |
| `v12_kaggle/` | Runs made in Environment 1 (Kaggle, NVIDIA T4) on the earlier v12 split: rebalancing, focal loss, the model-size sweep, RT-DETR, CBAM, WIoU, P2. Notebooks, per-class results, configurations and training logs. Kept as history: since v1.1.0 each of these variants has been retrained on v13, and the paper reports only the v13 runs. |

The folder names `v12_local` and `v12_kaggle` are historical: `v12_local` also holds all v13 work.

## The two splits

- **v13 (record).** 8,785 images (6,148 train, 1,758 validation, 879 test), 15,134 boxes, 22
  classes. No image, in any rotation or reflection, is shared between the training split and the
  validation or test split. All results of record in the paper are on v13. Files and runs carry
  the prefix `v13_`.
- **v12 (earlier split).** 17,136 images. It removed identical and re-compressed copies but not
  rotated or flipped ones. About two thirds of its validation and test images have such a copy in
  training. Only the configuration search, which chose the final system, rests on v12; every
  intervention reported in the paper was trained on v13. Files carry the prefix `v12_` or
  `local_`, and the Kaggle runs are in `v12_kaggle/`.

## Rebuilding the splits

1. Download the six source datasets in YOLO format from the URLs below.
2. Merge them with the notebooks in `dataset_preparation/`, adapting the paths. One step copied
   images to enlarge small classes; stage 1 removes those copies again.
3. Run `build_v12.py` (stage 1), then `v12_local/audit_dihedral_twins.py` on the result. That
   writes the twin-pair list `v12_local/results/dihedral_twins_pairs.csv`, also included here.
4. Run `build_v13.py` (stage 2).
5. Check the result against `v13_manifest.csv`.

The scripts keep the authors' absolute paths. Every dataset path starts with
`f:/0BScDefense/05_Datasets` (sometimes written with backslashes or a capital F). Replace that
prefix in the `.py` files with the folder that holds your datasets. Paths to results, runs and
tables are relative to each script and need no change.

Each image in the manifest has both a SHA-256 checksum and a perceptual hash (dHash). Roboflow may
re-encode images when it exports a dataset again. If it does, the checksum of a re-downloaded image
will differ, but its dHash will still match, so each image can still be assigned to its split.

## Training and evaluation

- **Training.** Each run is a notebook in `v12_local/nb/`, generated by `v12_local/build_nb.py`.
  `v12_local/run_v13_chain.py` runs the notebooks unattended. Every v13 run used Ultralytics YOLO
  with batch 16, mixed precision, 100 epochs and early-stopping patience 30, with two dataloader
  workers; the exact settings are in `runs/<run>/args.yaml`.
- **Models.** The P2 head is `v12_local/yolo11n_p2.yaml`. Build its runs with the arms `p2_rm`
  and `p2_wiou_rm` of `build_nb.py`. These map yolo11n's weights onto the P2 network by layer role
  (448 of 593 tensors), save the result as a checkpoint and train from it. A callback checks
  inside training that the weights arrived. The uncertainty-aware head (UADH) is
  `v12_local/uadh.py`.
- **Rebalancing.** `v12_local/build_resampled_v13.py` writes the oversampled and naive-resampled
  training sets (400 images per main class, seed 42) beside v13; validation and test stay
  unchanged.
- **Seeds.** Three-seed variants use seeds 0, 1 and 2; single-seed variants use seed 0.
- **Evaluation.** Test-split evaluation (`v12_local/nb/eval_v13.ipynb`, generated by
  `v12_local/build_eval_nb.py`) scores three seeds each of the baseline (YOLOv11n, 640 px),
  YOLOv11s at 1024 px, and YOLOv11s at 1024 px with test-time augmentation. It reports Welch's
  t-test, 95% Welch confidence intervals and Holm-adjusted p-values over five metrics. The output
  is `v12_local/results/eval_v13_*`.
  - `build_eval_nb.py --dataset v13_sub699` repeats the comparison on the 699 test images outside
    the v12 validation split.
  - `build_trainsplit_nb.py` scores the P2 runs on the training split.

Environment 2 was a workstation with an NVIDIA RTX 5070 Ti, PyTorch 2.13 (CUDA 13) and
Ultralytics 8.4. `requirements.txt` lists the Python packages.

## Rebuilding the tables

From `v12_local/`, in this order:

```
python analyse_protocol.py        # per-class label statistics (polygons read as boxes)
python analyse_object_scale.py    # COCO size bins
python build_table_dataset.py     # Table 1
python build_paper_assets.py      # Tables 2-5 and the figures
python build_extra_figs.py        # Figs. 1 and 5
python analyse_frequency.py       # class-frequency analysis
python build_reviewer_stats.py    # confound controls, per-seed results, robustness checks
python build_paper_tables_md.py   # paper/tables_generated.md
python manuscript_numbers_v13.py  # smaller numbers quoted in the text
python build_appendix_tables.py   # rows of Appendix Tables A.1-A.3
```

The scripts default to v13 and take `--dataset v12` for the earlier split, whose outputs carry the
prefix `v12_`. `manuscript_numbers_v13.py` is v13 only.

Group A (whole-leaf) and Group B (lesion) are defined in `v12_local/classes.py`. A class is in
Group B when its median box covers less than half of the image. The potato labels are polygons,
and `v12_local/labels.py` reads each one as its bounding box, as Ultralytics does.

## Source datasets

| Crop | Roboflow Universe dataset |
|---|---|
| Bean | https://universe.roboflow.com/object-detection-jr1kv/bean-leaf |
| Cauliflower | https://universe.roboflow.com/cauliflower-diseases/cauliflower-diseases-recognition |
| Chilli | https://universe.roboflow.com/vitap-university-kmcps/chilli-gliey |
| Cucumber | https://universe.roboflow.com/cucumber-bixgw/cucumber-leaf-h44fr |
| Potato | https://universe.roboflow.com/milagros-p6epw/potato-health-wxqiw |
| Tomato | https://universe.roboflow.com/appledetectionwithyolov8/tomato-leaf-dissease-detection |

## Licence

The code is released under the MIT licence (`LICENSE`). The manifest, distributions, result tables
and logs are released under CC BY 4.0 (`LICENSE-data.md`). The source images are not part of this
repository and keep their own licences.

## Citation

See `CITATION.cff`. Until the paper is published, please cite this repository.
