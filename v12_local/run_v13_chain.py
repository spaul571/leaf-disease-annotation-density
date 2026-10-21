"""Run the v13 notebooks one after another, headless, while the user is away.

Steps 3 and 4 of ../TWIN_LEAKAGE_PLAN.md. Started by Claude on 2026-10-21 at the user's request:
they are away, so the training notebooks are executed for them instead of from VS Code. Each
notebook runs in a fresh "Python (QML)" kernel through nbclient, so memory is released between
runs, and the executed copy (outputs included) is written back over the notebook, as VS Code would.

  * A training run counts as done when results/summary_<TAG>.json and its best.pt exist. Done
    runs are skipped, so after a fix the chain is simply started again.
  * After each run a sanity gate reads the result; if it fails, the chain stops and says why.
  * A watchdog kills a run whose results.csv stops growing, or whose loss turns NaN.
  * Windows is kept awake while the chain runs (SetThreadExecutionState), so an idle machine does
    not sleep through the campaign.
  * Every event goes to logs/v13_chain.log with a timestamp. Lines start with CHAIN, START,
    EPOCH, DONE, SKIP, FAIL or STALL, so the log is easy to filter.

    qpy run_v13_chain.py              # every core run not yet done, then nb/eval_v13.ipynb
    qpy run_v13_chain.py --no-eval    # core training only
    qpy run_v13_chain.py --plan arms  # the four intervention arms (logs/v13_arms.log), no eval
"""
import argparse
import asyncio
import csv
import ctypes
import json
import math
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import nbformat
import psutil
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
NB, RES, RUNS, LOGS = HERE / "nb", HERE / "results", HERE / "runs", HERE / "logs"
LOG = LOGS / "v13_chain.log"
EPOCHS = 100
GRACE_MIN = 45     # minutes allowed before the first epoch is written (caching, AMP check)
STALL_MIN = 30     # minutes allowed between two epochs (an epoch takes 1-3 min here)
MIN_MAP50 = 0.70   # validation mAP@50 below this means the run is broken, not merely worse

# (arm, seed, imgsz, workers), in run order. Recipe as the published v12 runs, except that every
# run uses 2 dataloader workers: with 8, the training and validation workers (8 + 16 processes at
# ~1.4 GB each) pushed commit charge to 58 of 60 GB on 2026-10-21. The R1-3 worker control found
# no effect of 8 vs 2 workers on accuracy, and 2 workers were faster here.
PLANS = {
    "core": [("baseline", s, 640, 2) for s in (0, 1, 2)] + [("small", s, 1024, 2) for s in (0, 1, 2)],
    # The local intervention arms, seed 0, retrained on v13 at the user's request on 2026-10-21
    # so that the intervention table no longer rests on leaked v12 data.
    "arms": [("p2", 0, 640, 2), ("p2_wiou", 0, 640, 2), ("uadh", 0, 640, 2),
             ("baseline", 0, 1024, 2)],
}


def names(arm, seed, imgsz, workers):
    """Run name, result tag and notebook path, mirroring the CONFIG cell of build_nb.py."""
    w = "" if workers == 8 else f"_W{workers}"
    sz = "" if imgsz == 640 else f"_{imgsz}"
    run = f"v13_{arm}_s{seed}_{imgsz}{w}"
    return run, f"v13_{arm}_{EPOCHS}ep_s{seed}{sz}{w}", NB / f"{run}.ipynb"


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:  # stdout may be a cp1252 file; never let a stray character kill the chain
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    except Exception:  # noqa: BLE001
        pass


def keep_awake(on):
    if os.name == "nt":
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            es_continuous | (es_system_required if on else 0))


def epochs_done(run):
    p = RUNS / run / "results.csv"
    if not p.exists():
        return 0, {}
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    return len(rows), ({k.strip(): v for k, v in rows[-1].items()} if rows else {})


def compact(nb):
    """Merge stream chunks and render carriage returns as a terminal would, so progress bars do
    not bloat the file."""
    for c in nb.cells:
        merged = []
        for o in c.get("outputs", []):
            if (o.get("output_type") == "stream" and merged
                    and merged[-1].get("output_type") == "stream"
                    and merged[-1]["name"] == o["name"]):
                merged[-1]["text"] += o["text"]
            else:
                merged.append(o)
        for o in merged:
            if o.get("output_type") == "stream":
                text = "\n".join(l.rsplit("\r", 1)[-1] for l in o["text"].split("\n"))
                if len(text) > 200_000:
                    text = (text[:20_000] + "\n[... trimmed by run_v13_chain.py ...]\n"
                            + text[-150_000:])
                o["text"] = text
        if c.get("cell_type") == "code":
            c["outputs"] = merged
    return nb


def execute(path, run=None):
    """Run one notebook in a fresh kernel and write it back. Returns (ok, reason)."""
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=None, startup_timeout=180, kernel_name="qml",
                            resources={"metadata": {"path": str(path.parent)}})
    state = {"finished": False, "stop": None}

    def kill_kernel(why):
        state["stop"] = why
        try:
            proc = psutil.Process(client.km.provisioner.pid)
            for child in proc.children(recursive=True):
                child.kill()
            proc.kill()
        except Exception as e:  # noqa: BLE001 - report, never crash the watchdog
            log(f"FAIL could not kill the kernel of {run}: {e!r}")

    def watchdog():
        t0 = last_t = time.time()
        last_n = 0
        while not state["finished"] and state["stop"] is None:
            time.sleep(60)
            if run is None or state["finished"]:
                continue
            n, row = epochs_done(run)
            now = time.time()
            if n > last_n:
                if n == 1 or n // 10 > last_n // 10:
                    log(f"EPOCH {run} {n}/{EPOCHS} mAP50 {float(row.get('metrics/mAP50(B)', 'nan')):.4f}"
                        f" mAP50-95 {float(row.get('metrics/mAP50-95(B)', 'nan')):.4f}"
                        f" ({(now - t0) / 60:.0f} min)")
                last_n, last_t = n, now
                losses = [float(row.get(k) or 0) for k in ("train/box_loss", "train/cls_loss")]
                if any(math.isnan(v) for v in losses):
                    log(f"STALL {run}: NaN training loss at epoch {n}")
                    kill_kernel("NaN training loss")
            elif (now - last_t) / 60 > (GRACE_MIN if last_n == 0 else STALL_MIN):
                log(f"STALL {run}: no new epoch for {(now - last_t) / 60:.0f} min (at epoch {n})")
                kill_kernel(f"stalled at epoch {n}")

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        client.execute()
        return True, ""
    except Exception as e:  # noqa: BLE001 - every failure is reported, the chain decides
        msg = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).strip().splitlines()
        return False, state["stop"] or f"{type(e).__name__}: {msg[-1] if msg else ''}"[:400]
    finally:
        state["finished"] = True
        nbformat.write(compact(nb), path)


def sanity(run, tag):
    s = RES / f"summary_{tag}.json"
    if not (RUNS / run / "weights" / "best.pt").exists():
        return False, "no best.pt"
    if not s.exists():
        return False, "no summary file (the report cell did not run)"
    m = json.loads(s.read_text())
    n, _ = epochs_done(run)
    if n < 31:
        return False, f"only {n} epochs"
    if not m["ALL"] >= MIN_MAP50:
        return False, f"validation mAP@50 {m['ALL']:.4f} is below {MIN_MAP50}"
    return True, (f"{n} epochs | val mAP50 {m['ALL']:.4f} | A {m['A_whole_leaf']:.4f}"
                  f" | B {m['B_lesion']:.4f} | recall {m['recall']:.4f}"
                  f" | mAP50-95 {m['mAP50_95']:.4f}")


def main(plan, with_eval):
    global LOG
    stem = "v13_chain" if plan == "core" else f"v13_{plan}"   # one log and pid file per plan
    LOG = LOGS / f"{stem}.log"
    LOGS.mkdir(exist_ok=True)
    pid_file = LOGS / f"{stem}.pid"
    pid_file.write_text(str(os.getpid()))
    keep_awake(True)
    log(f"CHAIN start ({plan}), pid {os.getpid()}")
    try:
        for arm, seed, imgsz, workers in PLANS[plan]:
            run, tag, path = names(arm, seed, imgsz, workers)
            if (RES / f"summary_{tag}.json").exists() and (RUNS / run / "weights" / "best.pt").exists():
                log(f"SKIP {run}: already done")
                continue
            if (RUNS / run).exists():  # a stopped or failed attempt: keep it aside, start clean
                old = RUNS / f"{run}_failed_{datetime.now():%Y%m%d_%H%M%S}"
                (RUNS / run).rename(old)
                log(f"moved the earlier attempt aside: {old.name}")
            log(f"START {run} ({path.name})")
            t0 = time.time()
            ok, why = execute(path, run)
            hours = (time.time() - t0) / 3600
            if not ok:
                log(f"FAIL {run} after {hours:.2f} h: {why}")
                return 1
            ok, why = sanity(run, tag)
            if not ok:
                log(f"FAIL {run} sanity check: {why}")
                return 2
            log(f"DONE {run} in {hours:.2f} h | {why}")
        if with_eval:
            path = NB / "eval_v13.ipynb"
            log(f"START eval_v13 ({path.name})")
            t0 = time.time()
            ok, why = execute(path)
            if not ok:
                log(f"FAIL eval_v13: {why}")
                return 3
            log(f"DONE eval_v13 in {(time.time() - t0) / 60:.0f} min")
        log("CHAIN COMPLETE")
        return 0
    finally:
        keep_awake(False)
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    if os.name == "nt":  # zmq needs add_reader, which the default Proactor loop lacks
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=sorted(PLANS), default="core")
    ap.add_argument("--no-eval", action="store_true", help="skip nb/eval_v13.ipynb at the end")
    a = ap.parse_args()
    # eval_v13.ipynb scores the core runs only; the arms are scored with the paper scripts
    raise SystemExit(main(a.plan, a.plan == "core" and not a.no_eval))
