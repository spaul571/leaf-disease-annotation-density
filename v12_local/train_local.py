"""Local (RTX 5070 Ti) runner for the v12 arms - same configs as the Kaggle kernels.

Purpose is NOT to re-run finished arms for a different answer; it is to
  (a) calibrate local hardware against the Kaggle T4 numbers (run --arm baseline --seed 0), and
  (b) get the multi-seed variance the campaign has never had (every Kaggle number is n=1).

Until (a) shows local baseline == 0.8975 +/- ~0.005, do NOT put local and Kaggle numbers in the
same table - hardware/cuDNN/AMP drift is the same size as the effects being reported.

Usage:
    qpy train_local.py --arm baseline --seed 0
    qpy train_local.py --arm p2       --seed 1
    qpy train_local.py --arm p2_wiou  --seed 1 --batch 12
"""
import argparse, csv, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "journal_analysis" / "v12_kaggle"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v12_kaggle"))

HERE = Path(__file__).resolve().parent
DATA = Path(r"f:\0BScDefense\05_Datasets\merged-dataset-v12\data.yaml")

# must match the Kaggle kernels EXACTLY or the group means are not comparable
NAMES = [
    "bean_angular_leaf_spot", "bean_bean_rust", "bean_healthy",
    "cauliflower_bacterial_spot_rot", "cauliflower_black_rot",
    "cauliflower_downy_mildew", "cauliflower_no_disease",
    "chilli_Curling-Leafs", "chilli_Healthy-Leafs", "chilli_Leaf Spot-Leafs",
    "chilli_Leveillula taurica-Leafs", "chilli_Mozaik Leaf",
    "cucumber_Antraknosa", "cucumber_Bacterial Wilt", "cucumber_Downy Mildew",
    "potato_healthy", "potato_blackleg", "potato_late_blight", "potato_early_blight",
    "tomato_bacterial_spot", "tomato_curl_virus", "tomato_two_spotted_spider_mite",
]
GROUP_B_LESION = {
    "potato_healthy", "cucumber_Antraknosa", "cucumber_Bacterial Wilt",
    "potato_late_blight", "potato_early_blight", "bean_bean_rust",
    "potato_blackleg", "bean_angular_leaf_spot",
}
group_of = lambda c: "B_lesion" if c in GROUP_B_LESION else "A_whole_leaf"

# Kaggle T4 reference (exp6b, 100 ep, seed 0) - the calibration target
REF = {"ALL": 0.8975, "A_whole_leaf": 0.9695, "B_lesion": 0.7716,
       "recall": 0.8697, "mAP50_95": 0.7083}


def build_model(arm):
    from ultralytics import YOLO
    import torch
    if arm == "baseline":
        return YOLO("yolo11n.pt")

    from build_p2 import P2_YAML  # import-safe: build loop is under __main__
    p2_path = HERE / "yolo11n_p2.yaml"
    p2_path.write_text(P2_YAML)
    model = YOLO(str(p2_path))
    pre = YOLO("yolo11n.pt").model.state_dict()
    st = model.model.state_dict()
    tx = 0
    for k, v in pre.items():
        if k in st and st[k].shape == v.shape:
            st[k] = v; tx += 1
    model.model.load_state_dict(st, strict=False)
    tot = sum(p.numel() for p in model.model.parameters())
    print(f"P2 built | params {tot/1e6:.3f}M | transferred {tx}")

    det = model.model.model[-1]
    strides = [int(s) for s in det.stride]
    assert det.nl == 4 and strides == [4, 8, 16, 32], f"P2 head wrong: nl={det.nl} {strides}"
    model.model.train()
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, 640, 640))
    feats = out["feats"] if isinstance(out, dict) else out
    assert len(feats) == 4 and tuple(feats[0].shape[-2:]) == (160, 160), "P2 smoke failed"
    print("P2 smoke OK:", [tuple(f.shape) for f in feats])
    return model


def apply_wiou():
    """Same WIoU v3 patch the exp19 kernel uses (single source of truth)."""
    from build_p2 import WIOU
    import torch, torch.nn as nn  # noqa: F401  (WIOU body references nn/torch)
    exec(WIOU, {"torch": torch, "nn": nn})


def val_perclass(weights, tag, split="val"):
    from ultralytics import YOLO
    b = YOLO(weights).val(data=str(DATA), split=split, imgsz=640, verbose=False).box
    rows = []
    for i, ci in enumerate(b.ap_class_index):
        ci = int(ci); name = NAMES[ci] if ci < len(NAMES) else str(ci)
        rows.append([tag, name, group_of(name), round(float(b.p[i]), 4), round(float(b.r[i]), 4),
                     round(float(b.ap50[i]), 4), round(float(b.ap[i]), 4)])
    rows.append([tag, "ALL", "ALL", round(float(b.mp), 4), round(float(b.mr), 4),
                 round(float(b.map50), 4), round(float(b.map), 4)])
    return rows


def report(rows, tag):
    print(f"\n=== {tag} ===")
    out = {}
    for g in ("A_whole_leaf", "B_lesion"):
        sel = [r for r in rows if r[2] == g]
        m = sum(r[5] for r in sel) / len(sel)
        out[g] = m
        print(f"{g:14s} mAP50 {m:.4f} ({m-REF[g]:+.4f} vs T4 baseline)  "
              f"recall {sum(r[4] for r in sel)/len(sel):.4f}")
    a = [r for r in rows if r[1] == "ALL"][0]
    out.update(ALL=a[5], recall=a[4], mAP50_95=a[6])
    print(f"{'ALL':14s} mAP50 {a[5]:.4f} ({a[5]-REF['ALL']:+.4f})  "
          f"recall {a[4]:.4f} ({a[4]-REF['recall']:+.4f})  "
          f"mAP50-95 {a[6]:.4f} ({a[6]-REF['mAP50_95']:+.4f})")
    print(f"A-B gap {out['A_whole_leaf']-out['B_lesion']:.4f} (T4 baseline 0.1979)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "p2", "p2_wiou"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    # exp6b/exp18/exp19 all used patience=30; keep it identical or calibration is invalid
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()

    import torch
    assert torch.cuda.is_available(), "no CUDA - install torch cu130 into the qml venv"
    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))

    if args.arm == "p2_wiou":
        apply_wiou()
    model = build_model(args.arm)

    run = f"local_{args.arm}_s{args.seed}_{args.imgsz}"
    tag = f"local_{args.arm}_{args.epochs}ep_s{args.seed}"
    model.train(data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=0, amp=True, patience=args.patience, seed=args.seed,
                workers=args.workers, project=str(HERE / "runs"), name=run, exist_ok=True)

    rows = val_perclass(HERE / "runs" / run / "weights" / "best.pt", tag)
    res_dir = HERE / "results"; res_dir.mkdir(exist_ok=True)
    csv_path = res_dir / f"perclass_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "class", "group", "precision", "recall", "mAP50", "mAP50_95"])
        w.writerows(rows)
    summary = report(rows, tag)
    (res_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print("SAVED", csv_path)


if __name__ == "__main__":
    main()
