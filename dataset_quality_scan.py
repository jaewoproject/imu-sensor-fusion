"""dataset_quality_scan.py — flag suspicious dataset/*.json files.

Use after recording (especially re-records of C-H) to catch corrupt samples
before training. Pre-empts the original DEBOUNCE bug where C-H all merged
into 1-stroke files without any indicator.

Flagged conditions (report only; nothing is deleted):
  - Too few total points (< 10)            → user clicked record but didn't write
  - Too many total points (> 500)          → user forgot to stop
  - Letters expected to be multi-stroke (E/F/H/T/X/Y) saved as 1 stroke
  - Single stroke shorter than 3 points    → likely accidental press
  - Any stroke with duplicate consecutive (x,y) — sensor stuck

Usage:
  py -3 dataset_quality_scan.py
"""

from __future__ import annotations

import glob
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

# Letters that nearly always need ≥ N strokes when block-printed by hand.
# Conservative — picks the floor, not the median.
MIN_STROKES = {
    "A": 2, "B": 2, "E": 3, "F": 3, "H": 3,
    "I": 1, "K": 2, "T": 2, "X": 2, "Y": 2,
}

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def scan_file(path: Path) -> list[str]:
    """Return list of human-readable warnings for this file. Empty = clean."""
    warnings = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return [f"unreadable: {e}"]

    label = data.get("label", path.stem.split("_")[0]).upper()
    strokes = data.get("strokes", [])
    if not strokes:
        return ["empty strokes"]

    n_strokes = len(strokes)
    n_pts = sum(len(s) for s in strokes)

    if n_pts < 10:
        warnings.append(f"only {n_pts} points (< 10 — likely empty press)")
    if n_pts > 500:
        warnings.append(f"{n_pts} points (> 500 — forgot to stop?)")

    min_req = MIN_STROKES.get(label)
    if min_req and n_strokes < min_req:
        warnings.append(
            f"{label} usually needs ≥{min_req} strokes, got {n_strokes} "
            f"(possible DEBOUNCE merge)"
        )

    for si, st in enumerate(strokes):
        if len(st) < 3:
            warnings.append(f"stroke {si}: only {len(st)} points")
            continue
        # Sensor-frozen check: ≥60% of consecutive (x,y) pairs identical means
        # the pen didn't move at all for most of the stroke. Natural pen pauses
        # produce ~10-30% duplicates which we tolerate.
        dups = sum(
            1 for a, b in zip(st, st[1:])
            if abs(a.get("x", 0) - b.get("x", 0)) < 1e-9
            and abs(a.get("y", 0) - b.get("y", 0)) < 1e-9
        )
        if dups / max(len(st) - 1, 1) >= 0.6:
            warnings.append(
                f"stroke {si}: {dups}/{len(st)-1} duplicate pairs ({dups/(len(st)-1)*100:.0f}% — sensor frozen?)"
            )

    return warnings


def main():
    root = Path(__file__).resolve().parent
    files = sorted(glob.glob(str(root / "dataset" / "*.json")))
    if not files:
        print("No dataset/*.json files found.")
        return

    print(f"Scanning {len(files)} files...\n")

    by_label = defaultdict(list)
    flagged = []
    for fp in files:
        path = Path(fp)
        label = path.stem.split("_")[0]
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            label = data.get("label", label).upper()
            n_strokes = len(data.get("strokes", []))
            n_pts = sum(len(s) for s in data.get("strokes", []))
            by_label[label].append((n_strokes, n_pts))
        except Exception:
            pass

        warns = scan_file(path)
        if warns:
            flagged.append((path.name, label, warns))

    # Per-label summary
    print("Per-label distribution:")
    print(f"  {'LBL':<4} {'N':>4}  {'stroke_count_dist':<35} {'points (min/med/max)'}")
    for lbl in sorted(by_label.keys()):
        rows = by_label[lbl]
        sc = defaultdict(int)
        for n, _ in rows:
            sc[n] += 1
        sc_str = ", ".join(f"{k}st:{v}" for k, v in sorted(sc.items()))
        pts = sorted(r[1] for r in rows)
        pmin, pmed, pmax = pts[0], pts[len(pts)//2], pts[-1]
        print(f"  {lbl:<4} {len(rows):>4}  {sc_str:<35} {pmin}/{pmed}/{pmax}")

    print(f"\nFlagged files: {len(flagged)} / {len(files)}")
    if flagged:
        for fname, label, warns in flagged:
            print(f"  [{label}] {fname}")
            for w in warns:
                print(f"      - {w}")
    else:
        print("  No issues detected.")

    # Actionable next-step recommendations
    print("\nReadiness check (≥30 samples per class recommended for training):")
    MIN_TRAIN = 30
    short = []
    for lbl in sorted(by_label.keys()):
        n = len(by_label[lbl])
        flag = "✓" if n >= MIN_TRAIN else "✗"
        print(f"  {flag} {lbl}: {n} samples {'' if n >= MIN_TRAIN else f'(need {MIN_TRAIN-n} more)'}")
        if n < MIN_TRAIN:
            short.append((lbl, MIN_TRAIN - n))

    if short:
        print("\nTo train: record the missing samples shown above, then:")
        print("  1) py -3 dataset_quality_scan.py    # re-verify")
        print("  2) py -u train_now.py               # retrain pure_bilstm")
        print("  3) py -3 verify_pipeline.py         # confirm 18/18 PASS + T5 live recognition")

    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
