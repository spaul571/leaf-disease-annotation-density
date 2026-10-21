"""Repair nb/local_p2_s0_640.ipynb in place.

The training cell is fine (155.7 min, 100/100 ep) and must NOT be re-run. Only the validation and
report cells are broken: their outputs were lost when the kernel was restarted, and re-running the
report cell alone raised `NameError: rows` because it depended on kernel state.

This swaps in the fixed cell sources (report now reads the CSV instead of memory), re-executes
just validation + report against the saved best.pt, and writes the real outputs back.
"""
import contextlib, io, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "nb" / "local_p2_s0_640.ipynb"
DONOR = HERE / "nb" / "local_p2_wiou_s0_640.ipynb"   # freshly built = has the fixed sources

nb = json.loads(TARGET.read_text(encoding="utf-8"))
donor = json.loads(DONOR.read_text(encoding="utf-8"))


def find(cells, prefix):
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "".join(c["source"]).startswith(prefix):
            return i
    raise SystemExit(f"cell starting {prefix!r} not found")


val_i, rep_i = find(nb["cells"], "# === Per-class"), find(nb["cells"], "# === Report")
nb["cells"][val_i]["source"] = donor["cells"][find(donor["cells"], "# === Per-class")]["source"]
nb["cells"][rep_i]["source"] = donor["cells"][find(donor["cells"], "# === Report")]["source"]

# rebuild the kernel state the two cells need, skipping the train cell entirely
g = {"__name__": "__main__"}
for prefix in ("# === Environment", "# === Config", "# === tqdm", "# === Model"):
    exec("".join(nb["cells"][find(nb["cells"], prefix)]["source"]), g)
g.setdefault("display", lambda x: None)

for idx, count in ((val_i, 6), (rep_i, 7)):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec("".join(nb["cells"][idx]["source"]), g)
    text = buf.getvalue()
    nb["cells"][idx]["outputs"] = [{"output_type": "stream", "name": "stdout",
                                    "text": text.splitlines(keepends=True)}]
    nb["cells"][idx]["execution_count"] = count
    print(f"--- cell {idx} ---\n{text}")

TARGET.write_text(json.dumps(nb, indent=1), encoding="utf-8")
errs = [o for c in nb["cells"] for o in c.get("outputs", []) if o.get("output_type") == "error"]
print("remaining errors in notebook:", len(errs))
