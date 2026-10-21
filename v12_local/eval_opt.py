"""Free-accuracy sweep on ALREADY-TRAINED weights: TTA and NMS threshold tuning.

No retraining. Every number is measured on the same clean v12 val split, so results are directly
comparable to the arms already in the table and can be judged against results/noise_floor.json.
"""
import argparse, json
from pathlib import Path
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATA = Path(r"f:\0BScDefense\05_Datasets\merged-dataset-v12\data.yaml")
GROUP_B = {"potato_healthy", "cucumber_Antraknosa", "cucumber_Bacterial Wilt",
           "potato_late_blight", "potato_early_blight", "bean_bean_rust",
           "potato_blackleg", "bean_angular_leaf_spot"}
NAMES = ["bean_angular_leaf_spot","bean_bean_rust","bean_healthy",
         "cauliflower_bacterial_spot_rot","cauliflower_black_rot","cauliflower_downy_mildew",
         "cauliflower_no_disease","chilli_Curling-Leafs","chilli_Healthy-Leafs",
         "chilli_Leaf Spot-Leafs","chilli_Leveillula taurica-Leafs","chilli_Mozaik Leaf",
         "cucumber_Antraknosa","cucumber_Bacterial Wilt","cucumber_Downy Mildew",
         "potato_healthy","potato_blackleg","potato_late_blight","potato_early_blight",
         "tomato_bacterial_spot","tomato_curl_virus","tomato_two_spotted_spider_mite"]


def ev(weights, imgsz, augment=False, iou=0.7, conf=0.001, tag=""):
    b = YOLO(str(weights)).val(data=str(DATA), split="val", imgsz=imgsz, verbose=False,
                               augment=augment, iou=iou, conf=conf,
                               project=str(HERE / "runs"), name="opt_tmp", exist_ok=True).box
    per = {}
    for i, ci in enumerate(b.ap_class_index):
        per[NAMES[int(ci)]] = float(b.ap50[i])
    gb = [v for k, v in per.items() if k in GROUP_B]
    ga = [v for k, v in per.items() if k not in GROUP_B]
    r = {"tag": tag, "ALL": round(float(b.map50), 4), "mAP50_95": round(float(b.map), 4),
         "recall": round(float(b.mr), 4),
         "A": round(sum(ga)/len(ga), 4), "B": round(sum(gb)/len(gb), 4)}
    print(f"{tag:38s} ALL {r['ALL']:.4f} | A {r['A']:.4f} | B {r['B']:.4f} | "
          f"R {r['recall']:.4f} | m50-95 {r['mAP50_95']:.4f}", flush=True)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="tta")
    a = ap.parse_args()
    W640 = HERE / "runs/local_baseline_s0_640/weights/best.pt"
    W1024 = HERE / "runs/local_baseline_s0_1024/weights/best.pt"
    out = []
    if a.mode == "tta":
        out.append(ev(W640, 640, tag="640 baseline (reference)"))
        out.append(ev(W640, 640, augment=True, tag="640 + TTA"))
        out.append(ev(W1024, 1024, tag="1024 baseline (reference)"))
        out.append(ev(W1024, 1024, augment=True, tag="1024 + TTA"))
    elif a.mode == "nms":
        for iou in (0.5, 0.6, 0.7, 0.8):
            out.append(ev(W1024, 1024, iou=iou, tag=f"1024 NMS iou={iou}"))
    (HERE / "results" / f"opt_{a.mode}.json").write_text(json.dumps(out, indent=2))
