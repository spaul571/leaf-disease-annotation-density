"""Measure 1024 throughput + RAM headroom. Must be a FILE, not stdin: Windows DataLoader
workers re-import __main__, and a stdin script makes every worker die with
OSError: Invalid argument: '<stdin>' - which looks exactly like a RAM crash but is not."""
import argparse, time, psutil, torch
from ultralytics import YOLO

st = {"n": 0, "t0": None, "peak": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=90)
    ap.add_argument("--weights", default="yolo11n.pt")
    a = ap.parse_args()

    def cb(tr):
        st["n"] += 1
        st["peak"] = max(st["peak"], psutil.virtual_memory().percent)
        if st["n"] == 10:
            st["t0"] = time.time()
        if st["n"] == a.iters:
            dt = time.time() - st["t0"]
            ips = (a.iters - 10) / dt
            iters = 11993 / a.batch
            print("RATE %s workers=%d imgsz=%d: %.2f it/s | epoch %.1f min | 100ep %.1f h | "
                  "peak RAM %.0f%% | VRAM %.1f GB"
                  % (a.weights, a.workers, a.imgsz, ips, iters / ips / 60, iters / ips / 3600 * 100,
                     st["peak"], torch.cuda.max_memory_reserved() / 1e9), flush=True)
            raise SystemExit(0)

    m = YOLO(a.weights)
    m.add_callback("on_train_batch_end", cb)
    m.train(data=r"f:\0BScDefense\05_Datasets\merged-dataset-v12\data.yaml", epochs=1,
            imgsz=a.imgsz, batch=a.batch, device=0, amp=True, workers=a.workers,
            val=False, plots=False, project="runs", name="rate_probe", exist_ok=True,
            verbose=False)


if __name__ == "__main__":
    main()
