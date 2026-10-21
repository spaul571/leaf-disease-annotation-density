"""Unattended chain runner: waits for the user's seed-2 run, then executes the remaining arms.

The user is AFK, so these are executed headlessly with nbconvert --inplace. Outputs are written
back INTO each .ipynb, so opening it later shows exactly what a manual Run All would have shown
(that is the point of keeping the notebook format even when nobody is watching).

Queue (each ~2.5-4h, strictly sequential - one GPU):
    1. nb/local_p2_s0_640.ipynb        P2 head, local env, seed 0
    2. nb/local_p2_wiou_s0_640.ipynb   P2 + WIoU, local env, seed 0

Why these two and not the 1024 arm: both are replications of arms already run on Kaggle into the
LOCAL environment, so they make the P2 effect quotable against the locally measured noise floor
without any new methodological commitment. The 1024 arm changes input resolution and breaks
comparability with the 640 ladder - that needs an explicit decision, so it is NOT queued here.

Start detached (survives the terminal closing):
    Start-Process -WindowStyle Hidden -FilePath "C:\\Users\\iZUMi\\.venvs\\qml\\Scripts\\python.exe" `
                  -ArgumentList "run_chain.py" -WorkingDirectory "<this dir>"
"""
import json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "logs"; LOG.mkdir(exist_ok=True)
LOGFILE = LOG / "chain.log"

WAIT_FOR = HERE / "results" / "summary_local_baseline_100ep_s2.json"
QUEUE = ["local_p2_s0_640.ipynb", "local_p2_wiou_s0_640.ipynb"]

SEED2_TIMEOUT_H = 6      # if seed 2 never lands, give up rather than hang forever
RAM_FREE_GB     = 7      # don't start a run while the previous kernel still holds memory
RAM_WAIT_MIN    = 45     # ...but don't wait forever for it either


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def free_gb():
    import psutil
    return psutil.virtual_memory().available / 1e9


def wait_for_seed2():
    if WAIT_FOR.exists():
        log(f"seed 2 already complete ({WAIT_FOR.name})")
        return True
    log(f"waiting for seed 2 -> {WAIT_FOR.name} (timeout {SEED2_TIMEOUT_H}h)")
    deadline = time.time() + SEED2_TIMEOUT_H * 3600
    last = 0
    while time.time() < deadline:
        if WAIT_FOR.exists():
            log("seed 2 complete")
            return True
        time.sleep(60)
        csv = HERE / "runs" / "local_baseline_s2_640" / "results.csv"
        if csv.exists():
            n = sum(1 for _ in open(csv)) - 1
            if n != last:
                log(f"  seed 2 progress: {n}/100 epochs")
                last = n
    log("TIMEOUT waiting for seed 2 - aborting chain")
    return False


def wait_for_ram():
    """The finished VS Code kernel can keep holding RAM; give it a chance to be released."""
    deadline = time.time() + RAM_WAIT_MIN * 60
    while free_gb() < RAM_FREE_GB and time.time() < deadline:
        log(f"  RAM only {free_gb():.1f} GB free (< {RAM_FREE_GB}); waiting for the previous "
            f"kernel to release. Restarting/closing that notebook's kernel would speed this up.")
        time.sleep(120)
    log(f"RAM {free_gb():.1f} GB free - proceeding")


def run_notebook(name):
    nb = HERE / "nb" / name
    if not nb.exists():
        log(f"MISSING {name} - skipping"); return False
    log(f"START {name}")
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
           "--inplace", "--ExecutePreprocessor.timeout=-1", str(nb)]
    out = LOG / f"{nb.stem}.log"
    with open(out, "w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(HERE))
    mins = (time.time() - t0) / 60
    if r.returncode == 0:
        log(f"DONE  {name} in {mins:.1f} min")
        for s in (HERE / "results").glob("summary_*.json"):
            if nb.stem.replace("local_", "").replace("_640", "") in s.stem:
                log(f"  {s.name}: {json.loads(s.read_text())}")
        return True
    log(f"FAILED {name} (exit {r.returncode}) after {mins:.1f} min - see logs/{nb.stem}.log")
    return False


if __name__ == "__main__":
    log("=" * 70)
    log(f"chain start | queue: {QUEUE}")
    if not wait_for_seed2():
        sys.exit(1)
    for name in QUEUE:
        wait_for_ram()
        ok = run_notebook(name)
        if not ok:
            log("continuing to next item despite failure")
    log("chain complete")
    log("NEXT: run nb/noise_floor.ipynb (needs seeds 0,1,2) and review the P2 results")
