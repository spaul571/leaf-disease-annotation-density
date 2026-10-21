"""Inputs of the paper scripts for one dataset, and the summaries and statistics they share.

v13 (v12 without its rotated/flipped twins, TWIN_LEAKAGE_PLAN.md) has been the dataset of record
since 2026-10-21. v12 is kept so the published numbers can still be rebuilt. Each paper script
takes --dataset (default v13), looks up its inputs with dataset(name), writes tables and figures
to D.out and result files to results/ with the prefix D.tag:

    v13 -> paper/       results/v13_*   (paper/ is what the manuscript build reads)
    v12 -> paper_v12/   results/v12_*

The assets behind the published v12 manuscript are frozen in paper_v12_published/, and the
unprefixed files in results/ (protocol_analysis.csv, reviewer_stats.json, ...) are their record.
Nothing here rewrites either.

Group means (A, B) are always recomputed from per-class values with classes.group_of. Files
written before 2026-10-21 store A/B under the old grouping, so their stored group fields are
ignored.

The Environment 1 arms (Kaggle era: resampling, focal loss, WIoU, the epoch-budget size sweep,
RT-DETR) exist on v12 only and are judged against their own v12 references and the v12 noise
floor, whichever dataset is the record.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats

from classes import group_of

HERE = Path(__file__).resolve().parent
JA = HERE.parent
RES = HERE / "results"
KAG = (JA / "v12_kaggle").resolve()
DS_ROOT = Path(r"F:\0BScDefense\05_Datasets")
METRICS = ("ALL", "A", "B", "recall", "mAP50_95")
TEST_SYSTEMS = ("baseline", "s@1024", "s@1024+TTA")


def _pc(name):
    return RES / f"perclass_{name}.csv"


DATASETS = {
    "v12": SimpleNamespace(
        name="v12", root=DS_ROOT / "merged-dataset-v12", dist=JA / "v12_distribution.csv",
        protocol=RES / "v12_protocol_analysis.csv", out=HERE / "paper_v12", tag="v12_",
        test_eval=RES / "twinfree_v12_eval.json",
        val_arms={   # arm -> per-class validation CSVs, one per seed (Environment 2, local)
            "baseline": [_pc(f"local_baseline_100ep_s{s}") for s in (0, 1, 2)],
            "+P2 head": [_pc("local_p2_100ep_s0")],
            "+P2+WIoU": [_pc("local_p2_wiou_100ep_s0")],
            "+UADH": [_pc(f"local_uadh_100ep_s{s}") for s in (0, 1)],
            "n @1024": [_pc("local_baseline_100ep_s0_1024")],
            "s @1024": [_pc(f"local_small_100ep_s{s}_1024") for s in (0, 1, 2)],
        }),
    "v13": SimpleNamespace(
        name="v13", root=DS_ROOT / "merged-dataset-v13", dist=JA / "v13_distribution.csv",
        protocol=RES / "v13_protocol_analysis.csv", out=HERE / "paper", tag="v13_",
        test_eval=RES / "eval_v13_eval.json",
        val_arms={   # every v13 run uses 2 dataloader workers (TWIN_LEAKAGE_PLAN.md, step 3)
            "baseline": [_pc(f"v13_baseline_100ep_s{s}_W2") for s in (0, 1, 2)],
            "+P2 head": [_pc("v13_p2_100ep_s0_W2")],
            "+P2+WIoU": [_pc("v13_p2_wiou_100ep_s0_W2")],
            "+UADH": [_pc("v13_uadh_100ep_s0_W2")],
            "n @1024": [_pc("v13_baseline_100ep_s0_1024_W2")],
            "s @1024": [_pc(f"v13_small_100ep_s{s}_1024_W2") for s in (0, 1, 2)],
        }),
}

# Environment 1 (Kaggle) arms on v12: arm -> (per-class CSV, reference arm)
ENV1 = {
    "baseline (Environment 1)": (KAG / "out6b/perclass_v12_100ep.csv", None),
    "oversample": (KAG / "out7/perclass_v12_oversample.csv", "baseline (Environment 1)"),
    "naive over+under": (KAG / "out14/perclass_v12_naive.csv", "baseline (Environment 1)"),
    "focal loss": (KAG / "out13/perclass_v12_focal.csv", "baseline (Environment 1)"),
    "+WIoU only": (KAG / "out17/perclass_v12_cbam_wiou.csv", "baseline (Environment 1)"),
    "n @60ep": (KAG / "out10/perclass_v12_nano60.csv", None),
    "s @60ep": (KAG / "out11/perclass_v12_small60.csv", "n @60ep"),
    "m @60ep": (KAG / "out12b/perclass_v12_medium60.csv", "n @60ep"),
    "RT-DETR-l @30ep": (KAG / "out15/perclass_v12_rtdetr.csv", None),
}


def dataset(name):
    return DATASETS[name]   # scripts that write create D.out themselves


def add_dataset_arg(ap):
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="v13",
                    help="v13 is the dataset of record; v12 rebuilds the published analysis")
    return ap


def images_on_disk(D):
    """Image files per split, the dataset totals quoted in the paper."""
    ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return {s: sum(1 for p in (D.root / "images" / s).iterdir() if p.suffix.lower() in ext)
            for s in ("train", "valid", "test")}


def perclass(path):
    """(ALL row, per-class rows) of a per-class CSV; `group` recomputed with classes.py."""
    d = pd.read_csv(path)
    per = d[d["class"] != "ALL"].copy()
    per["group"] = per["class"].map(group_of)
    return d[d["class"] == "ALL"].iloc[0], per


def summarise(path):
    """(metrics, per-class mAP@50) of one run. A and B are class means within each group."""
    a, per = perclass(path)
    g = per.groupby("group")["mAP50"].mean()
    return ({"ALL": float(a["mAP50"]), "A": float(g["A_whole_leaf"]), "B": float(g["B_lesion"]),
             "recall": float(a["recall"]), "mAP50_95": float(a["mAP50_95"])},
            per.set_index("class")["mAP50"])


def val_per_seed(D, arm="baseline"):
    """One row per seed of an arm on validation, with the A-B gap."""
    S = pd.DataFrame([summarise(f)[0] for f in D.val_arms[arm]])
    S["A_B_gap"] = S["A"] - S["B"]
    return S


def noise_floor(D):
    """Single-run significance bar per metric: 1.96 * sqrt(2) * sd over the baseline seeds."""
    S = val_per_seed(D)
    return {m: float(1.96 * np.sqrt(2) * S[m].std(ddof=1)) for m in S.columns}


def test_per_seed(D, subset="test_full"):
    """{system: DataFrame(seed x METRICS)} on the test split, with A and B regrouped."""
    E = [r for r in json.loads(D.test_eval.read_text()) if r["subset"] == subset]
    out = {}
    for system in TEST_SYSTEMS:
        rows = []
        for r in sorted((r for r in E if r["system"] == system), key=lambda r: r["seed"]):
            ga = [v for c, v in r["ap50"].items() if group_of(c) == "A_whole_leaf"]
            gb = [v for c, v in r["ap50"].items() if group_of(c) == "B_lesion"]
            rows.append({"seed": r["seed"], "ALL": r["ALL"], "A": float(np.mean(ga)),
                         "B": float(np.mean(gb)), "recall": r["recall"],
                         "mAP50_95": r["mAP50_95"]})
        assert len(rows) == 3, f"{D.test_eval.name}: {len(rows)} seeds of {system} in {subset}"
        out[system] = pd.DataFrame(rows).set_index("seed")
    return out


def test_bar(T):
    """Single-run bar per metric from the baseline seeds on the test split."""
    return {m: float(1.96 * np.sqrt(2) * T["baseline"][m].std(ddof=1)) for m in METRICS}


def welch_ci(a, b, alpha=0.05):
    """Welch confidence interval on mean(b) - mean(a), matching the Welch t-test in Table 5."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t = stats.t.ppf(1 - alpha / 2, df)
    d = b.mean() - a.mean()
    return float(d), float(d - t * se), float(d + t * se), float(df), float(t)


def holm(pvals):
    """Holm step-down adjustment of {name: p}; run it on unrounded p-values."""
    ps = sorted(pvals.items(), key=lambda kv: kv[1])
    adj, run_max = {}, 0.0
    for i, (k, p) in enumerate(ps):
        run_max = max(run_max, min((len(ps) - i) * p, 1.0))
        adj[k] = run_max
    return adj


def welch_vs_baseline(T):
    """{system: {metric: dict(delta, ci95, p, p_holm, ...)}} for every non-baseline system."""
    out = {}
    for system in TEST_SYSTEMS[1:]:
        res, raw = {}, {}
        for m in METRICS:
            a, b = T["baseline"][m].values, T[system][m].values
            d, lo, hi, dfree, tcrit = welch_ci(a, b)
            p = float(stats.ttest_ind(b, a, equal_var=False)[1])
            res[m] = {"delta": d, "ci95": [lo, hi], "welch_df": dfree, "t_crit": tcrit, "p": p,
                      "excludes_zero": bool(lo > 0 or hi < 0)}
            raw[m] = p
        for m, q in holm(raw).items():
            res[m]["p_holm"] = q
        out[system] = res
    return out
