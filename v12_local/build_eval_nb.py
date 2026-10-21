"""Generate the evaluation notebook for the twin-leakage plan (../TWIN_LEAKAGE_PLAN.md).

    qpy build_eval_nb.py --dataset v12     # step 1: re-score the published models on twin-free images
    qpy build_eval_nb.py --dataset v13     # step 4: score the v13 models (after they are trained)
    code -r nb/eval_twinfree_v12.ipynb

v12 mode scores the three published systems (baseline YOLOv11n 640, YOLOv11s 1024, YOLOv11s 1024
+ TTA; three seeds each) on the full test split and on two subsets built from the twin audit:
  twinfree - images with NO candidate twin in train (every candidate counts, so no leaked image
             can slip in);
  twinned  - images with a pixel-confirmed twin in train.
Images whose only twin is an unconfirmed candidate belong to neither subset. The baseline is also
scored on the matching validation subsets, which is where the density law is measured.

v13 mode scores the v13 runs on the full v13 test and validation splits.

No training happens here. Like every YOLO run in this project it is a notebook the user runs in
VS Code (live progress, kernel pinned to "Python (QML)"). Results are stored unrounded and the
evaluation is resumable: finished jobs are kept in results/<stem>_eval.json and skipped on re-run.
"""
import argparse
import json

from build_nb import ENV, HERE, code, md

SETTINGS = {
    "v12": {"data": r"f:\0BScDefense\05_Datasets\merged-dataset-v12",
            "base_run": "local_baseline_s{seed}_640", "small_run": "local_small_s{seed}_1024",
            "stem": "twinfree_v12", "nb": "eval_twinfree_v12.ipynb"},
    "v13": {"data": r"f:\0BScDefense\05_Datasets\merged-dataset-v13",
            "base_run": "v13_baseline_s{seed}_640_W2", "small_run": "v13_small_s{seed}_1024_W2",
            "stem": "eval_v13", "nb": "eval_v13.ipynb"},
}

HEADER = {
    "v12": """# Twin-free re-test of the published results (v12)

**Why.** About two thirds of the v12 validation and test images have a rotated or flipped copy of
themselves in the training split (`audit_dihedral_twins.py`, 2026-10-21). Every published test
number was measured on those images. This notebook re-scores the same nine model configurations
(baseline, YOLOv11s 1024, YOLOv11s 1024 + TTA; seeds 0-2) on images that have **no** twin in
train, and on images that do, without retraining anything.

**Questions it answers**
1. Does the headline result (YOLOv11s 1024 + TTA vs baseline: mAP@50-95 +0.032, p = 0.005; lesion
   group +0.036, p = 0.044) survive on twin-free test images?
2. Does the density law (per-class mAP@50 vs boxes per image, r = -0.921) survive on twin-free
   validation images?
3. How much do twinned images inflate the scores?

**Sanity check.** The full-test scores must reproduce the published ones (`test_final.json`,
`test_seeds_final.json`) to about 0.001. If they do not, stop: something in the setup differs.

**Runtime.** About 36 validation passes, roughly 30-60 minutes on the RTX 5070 Ti. The run is
resumable: re-running the evaluation cell skips finished jobs.

**When finished,** restart the kernel. Ultralytics' `val()` can leave dataloader workers alive.

Prerequisite: `results/dihedral_twins_pairs.csv` from `qpy audit_dihedral_twins.py`.
""",
    "v13": """# Test-split evaluation of the v13 runs

Scores the v13 runs (baseline YOLOv11n 640, YOLOv11s 1024, YOLOv11s 1024 + TTA; seeds 0-2) on
the full v13 test split, with Welch tests and Holm correction, and measures the density law on
the v13 validation split. This is the v13 counterpart of `eval_test.py` + `eval_test_seeds.py` +
`build_reviewer_stats.py` (R1-2).

Prerequisite: the six core v13 notebooks have finished (`nb/v13_baseline_s{0,1,2}_640_W2.ipynb`,
`nb/v13_small_s{0,1,2}_1024_W2.ipynb`).

**When finished,** restart the kernel. Ultralytics' `val()` can leave dataloader workers alive.
""",
}

CONFIG = r'''# === Config ===
import math
from collections import Counter, defaultdict
import numpy as np, pandas as pd, yaml
from scipy import stats

HERE    = Path(r"__HERE__")
DATASET = "__DATASET__"
DATA    = Path(r"__DATA__")          # dataset root, holds data.yaml
STEM    = "__STEM__"                 # results/<STEM>_*.json|csv|md
os.chdir(HERE)
sys.path.insert(0, str(HERE))
from eval_ensemble import NAMES, GROUP_B
from build_reviewer_stats import welch_ci

WORKERS  = 0      # dataloader workers per val(); 0 = none left behind in this long-lived kernel
MIN_INST = 5      # a class enters the density correlation only with >= 5 boxes in the subset
SEEDS    = (0, 1, 2)
SYSTEMS  = {      # system -> (run-name pattern, imgsz, TTA)
    "baseline":   ("__BASE_RUN__", 640, False),
    "s@1024":     ("__SMALL_RUN__", 1024, False),
    "s@1024+TTA": ("__SMALL_RUN__", 1024, True),
}
KEYS = ("ALL", "A", "B", "recall", "mAP50_95")
RES = HERE / "results"
SPL = HERE / "splits"; SPL.mkdir(exist_ok=True)


def weights(system, seed):
    return HERE / "runs" / SYSTEMS[system][0].format(seed=seed) / "weights" / "best.pt"


missing = [str(weights(s, k)) for s in SYSTEMS for k in SEEDS if not weights(s, k).exists()]
assert not missing, "missing weights:\n" + "\n".join(missing)
assert (DATA / "data.yaml").exists(), DATA
print(f"dataset {DATASET} | {len(SYSTEMS)} systems x {len(SEEDS)} seeds | weights found")
'''

SUBSETS = r'''# === Evaluation subsets ===
# name -> (data yaml, split argument for val(), image paths)
def images(split):
    return sorted(p for p in (DATA / "images" / split).iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


SUBSETS = {"test_full": (DATA / "data.yaml", "test", images("test")),
           "valid_full": (DATA / "data.yaml", "val", images("valid"))}

if DATASET == "v12":
    pairs = pd.read_csv(RES / "dihedral_twins_pairs.csv")

    def with_train_twin(confirmed_only):
        """(split, file) of every valid/test image whose twin group reaches into train."""
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        P = pairs[pairs.confirmed == 1] if confirmed_only else pairs
        for r in P.itertuples():
            parent[find((r.split_a, r.img_a))] = find((r.split_b, r.img_b))
        roots = {find(n) for n in list(parent) if n[0] == "train"}
        return {n for n in list(parent) if n[0] != "train" and find(n) in roots}

    lenient, strict = with_train_twin(False), with_train_twin(True)
    for split, arg in (("valid", "val"), ("test", "test")):
        imgs = images(split)
        for tag, keep in (("twinfree", lambda p: (split, p.name) not in lenient),
                          ("twinned", lambda p: (split, p.name) in strict)):
            name = f"{split}_{tag}"
            lst = [p for p in imgs if keep(p)]
            txt = SPL / f"v12_{name}.txt"
            txt.write_text("\n".join(str(p) for p in lst) + "\n", encoding="utf-8")
            yml = SPL / f"v12_{name}.yaml"
            yml.write_text(yaml.safe_dump(
                {"path": str(DATA), "train": "images/train", "val": str(txt), "test": str(txt),
                 "nc": len(NAMES), "names": NAMES}, sort_keys=False), encoding="utf-8")
            SUBSETS[name] = (yml, "test", lst)

# boxes per class in each subset, to flag classes too thin to score
INST = {}
for name, (_, _, lst) in SUBSETS.items():
    c = Counter()
    for p in lst:
        lbl = p.parent.parent.parent / "labels" / p.parent.name / (p.stem + ".txt")
        if lbl.exists():
            c.update(NAMES[int(l.split()[0])] for l in lbl.read_text().splitlines() if l.strip())
    INST[name] = dict(c)
inst = pd.DataFrame(INST).fillna(0).astype(int).reindex(NAMES)
inst.loc["images"] = [len(SUBSETS[n][2]) for n in inst.columns]
inst.to_csv(RES / f"{STEM}_subset_boxes.csv")
display(inst)
'''

EVAL = r'''# === Evaluate (resumable) ===
from ultralytics import YOLO

JOBS = [(sub, system, seed) for sub in SUBSETS for system in SYSTEMS for seed in SEEDS
        if sub.startswith("test") or system == "baseline"]
out_json = RES / f"{STEM}_eval.json"
done = json.loads(out_json.read_text()) if out_json.exists() else []
have = {(r["subset"], r["system"], r["seed"]) for r in done}
todo = [j for j in JOBS if j not in have]
print(f"{len(JOBS)} jobs, {len(todo)} still to run")

bar = tqdm(todo, unit="job", file=sys.stdout, dynamic_ncols=True)
for sub, system, seed in bar:
    bar.set_description(f"{sub} | {system} s{seed}")
    _, imgsz, tta = SYSTEMS[system]
    yml, split, lst = SUBSETS[sub]
    b = YOLO(str(weights(system, seed))).val(
        data=str(yml), split=split, imgsz=imgsz, augment=tta, workers=WORKERS, plots=False,
        verbose=False, project=str(HERE / "runs"), name="eval_tmp", exist_ok=True).box
    ap50 = {NAMES[int(c)]: float(b.ap50[i]) for i, c in enumerate(b.ap_class_index)}
    ap95 = {NAMES[int(c)]: float(b.ap[i]) for i, c in enumerate(b.ap_class_index)}
    ga = [v for k, v in ap50.items() if k not in GROUP_B]
    gb = [v for k, v in ap50.items() if k in GROUP_B]
    done.append({"subset": sub, "system": system, "seed": seed, "n_images": len(lst),
                 "ALL": float(b.map50), "A": float(np.mean(ga)) if ga else math.nan,
                 "B": float(np.mean(gb)) if gb else math.nan, "recall": float(b.mr),
                 "mAP50_95": float(b.map), "n_classes": len(ap50), "ap50": ap50, "ap50_95": ap95})
    out_json.write_text(json.dumps(done, indent=1))      # unrounded; saved after every job
    bar.set_postfix({"mAP50": f"{float(b.map50):.4f}",
                     "RAM%": f"{psutil.virtual_memory().percent:.0f}"})
    gc.collect(); torch.cuda.empty_cache()
bar.close()
R = pd.DataFrame(done)
display(R.groupby(["subset", "system"])[list(KEYS)].mean().round(4))
'''

SANITY = r'''# === Sanity: the full test split must reproduce the published numbers ===
pub = {("baseline", int(r["tag"][-1])): r
       for r in json.loads((RES / "test_final.json").read_text())
       if r["tag"].startswith("640 baseline")}
seeds = json.loads((RES / "test_seeds_final.json").read_text())
for system, key in (("s@1024", "arm"), ("s@1024+TTA", "arm_tta")):
    for r in seeds[key]:
        pub[(system, int(r["tag"][-1]))] = r
worst = 0.0
for r in R[R.subset == "test_full"].itertuples():
    p = pub[(r.system, r.seed)]
    worst = max(worst, *(abs(getattr(r, k) - p[k]) for k in KEYS))
print(f"largest |re-run - published| on the full test split: {worst:.4f}")
print("OK - the setup reproduces the published evaluation" if worst <= 0.0015 else
      "STOP - the full-test scores differ from the published ones; find out why before "
      "reading anything below")
'''

STATS = r'''# === Welch tests: each system vs the baseline, per test subset ===
def holm(ps):
    adj, run = {}, 0.0
    for i, k in enumerate(sorted(ps, key=ps.get)):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        adj[k] = run
    return adj


STATS = {}
for sub in [s for s in SUBSETS if s.startswith("test")]:
    T = R[R.subset == sub]
    base = T[T.system == "baseline"].sort_values("seed")
    STATS[sub] = {}
    print(f"\n=== {sub}: {len(SUBSETS[sub][2])} images ===")
    for system in [s for s in SYSTEMS if s != "baseline"]:
        arm = T[T.system == system].sort_values("seed")
        rows, ps = {}, {}
        for k in KEYS:
            d, lo, hi, dfree, _ = welch_ci(base[k].values, arm[k].values)
            p = float(stats.ttest_ind(arm[k].values, base[k].values, equal_var=False).pvalue)
            ps[k] = p
            rows[k] = {"baseline": float(base[k].mean()), "baseline_sd": float(base[k].std(ddof=1)),
                       "system": float(arm[k].mean()), "system_sd": float(arm[k].std(ddof=1)),
                       "delta": d, "ci95": [lo, hi], "welch_df": dfree, "p": p}
        for k, a in holm(ps).items():
            rows[k]["p_holm"] = a
        STATS[sub][system] = rows
        print(f"  {system}")
        for k in KEYS:
            r = rows[k]
            print(f"    {k:9s} {r['baseline']:.3f} -> {r['system']:.3f}   {r['delta']:+.3f} "
                  f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]   p {r['p']:.3f}   Holm {r['p_holm']:.3f}")
_ = (RES / f"{STEM}_stats.json").write_text(json.dumps(STATS, indent=1))
'''

LEAK = r'''# === How much do twinned images inflate the score? ===
# Raw subset means mix two things: leakage, and class mix (the twinned subset is mostly tomato and
# cucumber). The per-class comparison holds the class fixed: same class, twinned vs twin-free.
LEAK = {}
for system in SYSTEMS:
    S = R[(R.system == system) & R.subset.isin(["test_twinned", "test_twinfree"])]
    raw = S.groupby("subset")[list(KEYS)].mean()
    tw = pd.DataFrame(list(S[S.subset == "test_twinned"].ap50)).mean()
    fr = pd.DataFrame(list(S[S.subset == "test_twinfree"].ap50)).mean()
    both = [c for c in NAMES if INST["test_twinned"].get(c, 0) >= MIN_INST
            and INST["test_twinfree"].get(c, 0) >= MIN_INST]
    per = (tw[both] - fr[both]).sort_values()
    LEAK[system] = {"raw": (raw.loc["test_twinned"] - raw.loc["test_twinfree"]).to_dict(),
                    "per_class": per.to_dict(), "per_class_mean": float(per.mean())}
    print(f"{system:11s} raw twinned - twin-free: "
          + "  ".join(f"{k} {LEAK[system]['raw'][k]:+.3f}" for k in KEYS))
    print(f"{'':11s} same class, twinned - twin-free AP@50: mean {per.mean():+.3f} over "
          f"{len(both)} classes (" + ", ".join(f"{c} {v:+.3f}" for c, v in per.items()) + ")")
_ = (RES / f"{STEM}_leakage.json").write_text(json.dumps(LEAK, indent=1))
'''

DENSITY = r'''# === Density law: per-class mAP@50 of the baseline vs boxes per image ===
# boxes per image = mean number of a class's boxes over the images containing it, all splits
# pooled (the definition of analyse_protocol.py, recomputed here from this dataset's labels).
per = defaultdict(list)
for split in ("train", "valid", "test"):
    for lbl in (DATA / "labels" / split).glob("*.txt"):
        c = Counter(int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip())
        for k, n in c.items():
            per[NAMES[k]].append(n)
DENS = {k: float(np.mean(v)) for k, v in per.items()}

rows = []
for sub in [s for s in ("valid_full", "valid_twinfree", "test_full", "test_twinfree") if s in SUBSETS]:
    B = R[(R.subset == sub) & (R.system == "baseline")]
    ap = pd.DataFrame(list(B.ap50)).mean()
    cls = [c for c in ap.index if INST[sub].get(c, 0) >= MIN_INST]
    x = np.array([DENS[c] for c in cls]); y = ap[cls].values
    r, p = stats.pearsonr(x, y)
    rho, prho = stats.spearmanr(x, y)
    rows.append({"subset": sub, "classes": len(cls), "pearson_r": r, "p": p,
                 "spearman_rho": rho, "p_spearman": prho,
                 "dropped_thin_classes": [c for c in ap.index if c not in cls]})
    print(f"{sub:15s} {len(cls):2d} classes   r = {r:+.3f} (p = {p:.1e})   "
          f"Spearman {rho:+.3f} (p = {prho:.1e})")
DENSITY = pd.DataFrame(rows)
DENSITY.to_csv(RES / f"{STEM}_density.csv", index=False)
'''

VERDICT = r'''# === Verdict ===
def claim(sub, system, k):
    r = STATS[sub][system][k]
    lo, hi = r["ci95"]
    ok = r["p"] < 0.05 and (lo > 0 or hi < 0)
    return (f"{system} vs baseline, {k}: {r['delta']:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], "
            f"p = {r['p']:.3f}, Holm p = {r['p_holm']:.3f} -> "
            + ("significant" + ("" if r["p_holm"] < 0.05 else ", but not after Holm")
               if ok else "NOT significant"))

lines = [f"# {STEM}: summary", ""]
main = "test_twinfree" if "test_twinfree" in STATS else "test_full"
for sub in [s for s in ("test_full", "test_twinfree", "test_twinned") if s in STATS]:
    lines += [f"## {sub} ({len(SUBSETS[sub][2])} images)", ""]
    lines += [f"- {claim(sub, 's@1024+TTA', k)}" for k in ("mAP50_95", "B", "ALL")]
    lines += [f"- {claim(sub, 's@1024', k)}" for k in ("mAP50_95", "B")]
    lines.append("")
lines += ["## Density law (baseline, per-class mAP@50 vs boxes per image)", ""]
lines += [f"- {r.subset}: r = {r.pearson_r:+.3f} (p = {r.p:.1e}), {r.classes} classes"
          for r in DENSITY.itertuples()]
if DATASET == "v12":
    lines += ["", "## Leakage effect (test, same class, twinned - twin-free AP@50)", ""]
    lines += [f"- {s}: {v['per_class_mean']:+.3f}" for s, v in LEAK.items()]
text = "\n".join(lines) + "\n"
(RES / f"{STEM}_summary.md").write_text(text, encoding="utf-8")
print(text)
print(f"headline subset for the paper: {main}")
'''

FOOTER = """## Next

Take the summary above to `../TWIN_LEAKAGE_PLAN.md`, decision gate after step 1 (v12) or step 4
(v13). Then restart the kernel.
"""


def build(dataset, force):
    s = SETTINGS[dataset]
    sub = lambda t: (t.replace("__HERE__", str(HERE)).replace("__DATASET__", dataset)
                      .replace("__DATA__", s["data"]).replace("__STEM__", s["stem"])
                      .replace("__BASE_RUN__", s["base_run"]).replace("__SMALL_RUN__", s["small_run"]))
    cells = [md(HEADER[dataset]), code(ENV), code(sub(CONFIG)), code(SUBSETS), code(EVAL)]
    if dataset == "v12":
        cells.append(code(SANITY))
    cells.append(code(STATS))
    if dataset == "v12":
        cells.append(code(LEAK))
    cells += [code(DENSITY), code(VERDICT), md(FOOTER)]
    nbjson = {"cells": cells,
              "metadata": {"kernelspec": {"display_name": "Python (QML)", "language": "python",
                                          "name": "qml"},
                           "language_info": {"name": "python", "version": "3.12.13"}},
              "nbformat": 4, "nbformat_minor": 5}
    path = HERE / "nb" / s["nb"]
    if path.exists() and not force:
        old = json.loads(path.read_text(encoding="utf-8"))
        if any(c.get("outputs") for c in old.get("cells", [])):
            raise SystemExit(f"REFUSING to overwrite {path.name}: it holds executed outputs "
                             f"(pass --force to regenerate)")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(nbjson, indent=1), encoding="utf-8")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(SETTINGS), default="v12")
    ap.add_argument("--force", action="store_true", help="overwrite a notebook holding outputs")
    a = ap.parse_args()
    p = build(a.dataset, a.force)
    print("built", p)
    print(f'open with:  code -r "{p}"')
