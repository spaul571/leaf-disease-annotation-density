"""Read YOLO label lines that may be boxes or polygons.

The potato source dataset is labelled with polygons (`class x1 y1 x2 y2 ...`), not boxes
(`class xc yc w h`): 5,212 of the 28,530 label lines in v12. Ultralytics turns each polygon into
its bounding box before training and evaluation, so every script that reads label geometry must do
the same. Reading columns 3-4 of a polygon line as width and height is the bug that corrupted the
potato box areas (TWIN_LEAKAGE_PLAN.md, section 1b).

One difference to keep in mind: Ultralytics treats a whole file as polygons as soon as one line in
it has more than 6 fields. A file that mixed box and polygon lines would then have its box lines
misread. `parse_line` reads each line on its own, so callers that care should count mixed files
(analyse_protocol.py does) and confirm there are none.
"""
from pathlib import Path


def parse_line(line):
    """(class, xc, yc, w, h, is_polygon) in normalised units, or None for a blank/bad line."""
    p = line.split()
    if len(p) == 5:
        return int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), False
    if len(p) >= 7 and len(p) % 2 == 1:
        xs, ys = [float(v) for v in p[1::2]], [float(v) for v in p[2::2]]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        return int(p[0]), (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, True
    return None


def read_boxes(path):
    """[(class, xc, yc, w, h)] for one label file; a polygon becomes its bounding box."""
    out = []
    for line in Path(path).read_text().splitlines():
        r = parse_line(line)
        if r is not None:
            out.append(r[:5])
    return out
