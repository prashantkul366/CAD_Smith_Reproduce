"""Check each stored reference STL against the entry's own reference code.

The references are ground truth: every CD, F1 and IoU in every experiment is
measured against them. A stale or mismatched file does not announce itself -
it shows up as one entry with an enormous chamfer distance, which reads as a
model failure and drags the tier mean with it. T1_031 was stored as a
60x60x100 solid while its own reference code builds a 50x30x40 frustum.

Rebuilds each reference from `reference_code` and compares the bounding box to
the stored STL. Needs no credentials and no network.

    python scripts/check_reference_stls.py            # all tiers
    python scripts/check_reference_stls.py --tiers T1
"""

import argparse
import json
import struct
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "dataset_v2"
STL_DIR = DATA_DIR / "reference_stls"
TIER_FILES = {
    "T1": "t1_primitives.jsonl",
    "T2": "t2_engineering_parts.jsonl",
    "T3": "t3_complex_parts.jsonl",
}

#: A stored mesh is a triangulation of a curved solid, so it sits slightly
#: inside the analytic surface. This is the slack that allows for.
TOLERANCE_MM = 0.25


def stl_bbox(path: Path):
    """Bounding box of a binary STL, without pulling in a mesh library."""
    raw = path.read_bytes()
    count = struct.unpack("<I", raw[80:84])[0]
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    off = 84
    for _ in range(count):
        tri = struct.unpack("<12fH", raw[off:off + 50])
        off += 50
        for corner in range(3):
            for axis in range(3):
                v = tri[3 + 3 * corner + axis]
                lo[axis] = min(lo[axis], v)
                hi[axis] = max(hi[axis], v)
    return [hi[i] - lo[i] for i in range(3)]


def built_bbox(code: str, executor):
    result = executor.execute(code, name="ref_check")
    if not result.success:
        return None, (result.error or "")[:90]
    bb = (result.geometry_json or {}).get("bounding_box") or {}
    if not bb:
        return None, "no bounding box reported"
    return [bb["xlen"], bb["ylen"], bb["zlen"]], None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiers", nargs="+", default=["T1", "T2", "T3"])
    args = ap.parse_args()

    from autofab.executor import Executor
    executor = Executor(output_dir=tempfile.mkdtemp(prefix="refcheck_"),
                        timeout_seconds=300)

    checked = mismatched = unbuildable = missing = 0
    bad = []

    for tier in args.tiers:
        path = DATA_DIR / TIER_FILES[tier]
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            eid = entry["id"]
            stl = STL_DIR / f"{eid}.stl"
            if not stl.exists():
                print(f"  MISSING  {eid}: no reference STL")
                missing += 1
                continue

            code = entry.get("reference_code") or entry.get("code")
            want, err = built_bbox(code, executor)
            checked += 1
            if want is None:
                print(f"  UNBUILT  {eid}: reference code fails - {err}")
                unbuildable += 1
                continue

            got = stl_bbox(stl)
            delta = [abs(a - b) for a, b in zip(got, want)]
            if max(delta) > TOLERANCE_MM:
                print(f"  MISMATCH {eid}: stl "
                      f"{got[0]:.2f} x {got[1]:.2f} x {got[2]:.2f}  vs  code "
                      f"{want[0]:.2f} x {want[1]:.2f} x {want[2]:.2f}")
                mismatched += 1
                bad.append(eid)

    print()
    print(f"  checked            : {checked}")
    print(f"  bbox mismatches    : {mismatched}")
    print(f"  reference unbuilt  : {unbuildable}")
    print(f"  missing STLs       : {missing}")
    if bad:
        print(f"\n  Every metric for these entries is measured against the wrong")
        print(f"  solid: {', '.join(bad)}")
        print(f"  Regenerate with: python scripts/generate_reference_stls.py")
    return 1 if (mismatched or unbuildable or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
