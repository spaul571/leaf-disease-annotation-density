"""Generate nb/v13_trainsplit_p2.ipynb: the P2 diagnostic Shakib asked for (S-6).

Scores the three baseline seeds and the four P2 runs (from scratch and with pretrained weights, each
with and without WIoU) on the TRAINING split and on the validation split, with the same validation
code, and reports class-mean, Group A and Group B mAP@50, precision and recall for both. Overfitting
shows as a training-split score above the baseline's with a validation score below it; failure to
fit shows as a training-split score below the baseline's. No training happens here.

    qpy build_trainsplit_nb.py
    qpy run_v13_chain.py --plan trainsplit      # or run the notebook in VS Code
"""
import json

from build_nb import ENV, HERE, code, md

HEADER = """# P2 diagnostic: training split against validation split (Shakib S-6)

Scores the baseline seeds and the four P2 runs on the training split and the validation split of
v13, with the same evaluation code. Results: `results/v13_trainsplit_eval.json` (unrounded, per
class) and `results/v13_trainsplit_summary.csv`. Resumable: finished jobs are skipped.

**When finished,** restart the kernel. Ultralytics' `val()` can leave dataloader workers alive.
"""

EVAL = r'''# === Evaluate on the training and validation splits (resumable) ===
import json
import numpy as np, pandas as pd
from ultralytics import YOLO

HERE = Path(r"__HERE__")
os.chdir(HERE)
sys.path.insert(0, str(HERE))
from classes import NAMES, group_of

DATA = Path(r"f:\0BScDefense\05_Datasets\merged-dataset-v13\data.yaml")
RUNS = {"baseline s0": "v13_baseline_s0_640_W2", "baseline s1": "v13_baseline_s1_640_W2",
        "baseline s2": "v13_baseline_s2_640_W2",
        "P2 from scratch": "v13_p2_s0_640_W2", "P2+WIoU from scratch": "v13_p2_wiou_s0_640_W2",
        "P2 pretrained": "v13_p2_rm_s0_640_W2", "P2+WIoU pretrained": "v13_p2_wiou_rm_s0_640_W2"}
SPLITS = ("train", "val")
out_json = HERE / "results" / "v13_trainsplit_eval.json"
done = json.loads(out_json.read_text()) if out_json.exists() else []
have = {(r["run"], r["split"]) for r in done}
jobs = [(n, s) for n in RUNS for s in SPLITS if (n, s) not in have]
print(f"{len(RUNS) * len(SPLITS)} jobs, {len(jobs)} still to run")
bar = tqdm(jobs, unit="job", file=sys.stdout, dynamic_ncols=True)
for name, split in bar:
    bar.set_description(f"{name} | {split}")
    w = HERE / "runs" / RUNS[name] / "weights" / "best.pt"
    b = YOLO(str(w)).val(data=str(DATA), split=split, imgsz=640, workers=0, plots=False,
                         verbose=False, project=str(HERE / "runs"), name="eval_tmp",
                         exist_ok=True).box
    per = {NAMES[int(c)]: {"precision": float(b.p[i]), "recall": float(b.r[i]),
                           "ap50": float(b.ap50[i]), "ap50_95": float(b.ap[i])}
           for i, c in enumerate(b.ap_class_index)}
    done.append({"run": name, "split": split, "map50": float(b.map50), "map50_95": float(b.map),
                 "precision": float(b.mp), "recall": float(b.mr), "per_class": per})
    out_json.write_text(json.dumps(done, indent=1))
    gc.collect(); torch.cuda.empty_cache()
bar.close()
'''

SUMMARY = r'''# === Summary by group ===
rows = []
for r in done:
    per = pd.DataFrame(r["per_class"]).T
    per["group"] = [group_of(c) for c in per.index]
    g = per.groupby("group")[["ap50", "precision", "recall"]].mean()
    rows.append({"run": r["run"], "split": r["split"], "class_mean_ap50": per["ap50"].mean(),
                 "A_ap50": g.loc["A_whole_leaf", "ap50"], "B_ap50": g.loc["B_lesion", "ap50"],
                 "B_precision": g.loc["B_lesion", "precision"], "B_recall": g.loc["B_lesion", "recall"],
                 "map50_95": r["map50_95"]})
S = pd.DataFrame(rows)
S.to_csv(HERE / "results" / "v13_trainsplit_summary.csv", index=False)
wide = S.pivot(index="run", columns="split", values=["class_mean_ap50", "B_ap50"]).round(4)
wide[("gap", "class_mean")] = (wide[("class_mean_ap50", "train")] - wide[("class_mean_ap50", "val")]).round(4)
wide[("gap", "B")] = (wide[("B_ap50", "train")] - wide[("B_ap50", "val")]).round(4)
display(wide.reindex(list(RUNS)))
'''


def build():
    cells = [md(HEADER), code(ENV), code(EVAL.replace("__HERE__", str(HERE))), code(SUMMARY)]
    nbjson = {"cells": cells,
              "metadata": {"kernelspec": {"display_name": "Python (QML)", "language": "python",
                                          "name": "qml"},
                           "language_info": {"name": "python", "version": "3.12.13"}},
              "nbformat": 4, "nbformat_minor": 5}
    path = HERE / "nb" / "v13_trainsplit_p2.ipynb"
    if path.exists() and any(c.get("outputs") for c in
                             json.loads(path.read_text(encoding="utf-8")).get("cells", [])):
        raise SystemExit(f"REFUSING to overwrite {path.name}: it holds executed outputs")
    path.write_text(json.dumps(nbjson, indent=1), encoding="utf-8")
    return path


if __name__ == "__main__":
    print("built", build())
