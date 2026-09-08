"""Check each stored reference STL against the entry's own reference code.

The references are ground truth: every CD, F1 and IoU in every experiment is
measured against them. A stale or mismatched file does not announce itself -
it shows up as one entry with an enormous chamfer distance, which reads as a
model failure and drags the tier mean with it. T1_031 was stored as a
60x60x100 solid while its own reference code builds a 50x30x40 frustum.

Rebuilds each reference from `reference_code` and compares the bounding box to
the stored STL. Needs no credentials and no network.

Nothing else in the repo rebuilds these files - generate_reference_stls.py
works on data_test.jsonl, a different dataset - so --fix rewrites a mismatched
reference from its own code. It only ever touches a file that is provably
wrong, and it invalidates any recorded result for that entry: the metrics in
results.jsonl were measured against the old solid and have to be recomputed.

    python scripts/check_reference_stls.py            # report
    python scripts/check_reference_stls.py --fix      # rewrite the mismatches
"""

import argparse
import json
import shutil
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


def build(code: str, executor, name: str):
    """Build a reference and return (bbox, stl path, error)."""
    result = executor.execute(code, name=name)
    if not result.success:
        return None, None, (result.error or "")[:90]
    bb = (result.geometry_json or {}).get("bounding_box") or {}
    if not bb:
        return None, None, "no bounding box reported"
    return [bb["xlen"], bb["ylen"], bb["zlen"]], result.stl_path, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiers", nargs="+", default=["T1", "T2", "T3"])
    ap.add_argument("--fix", action="store_true",
                    help="rewrite mismatched references from their own code")
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
            want, built_stl, err = build(code, executor, eid)
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
                if args.fix and built_stl:
                    backup = stl.with_suffix(".stl.bak")
                    if not backup.exists():
                        shutil.copy(stl, backup)
                    shutil.copy(built_stl, stl)
                    print(f"           rewritten from reference_code "
                          f"(old file kept at {backup.name})")

    print()
    print(f"  checked            : {checked}")
    print(f"  bbox mismatches    : {mismatched}")
    print(f"  reference unbuilt  : {unbuildable}")
    print(f"  missing STLs       : {missing}")
    if bad and not args.fix:
        print(f"\n  Every metric for these entries is measured against the wrong")
        print(f"  solid: {', '.join(bad)}")
        print(f"  Rewrite them from their own code with --fix.")
    elif bad:
        print(f"\n  Rewrote {len(bad)}: {', '.join(bad)}")
        print(f"  Any recorded result for these was scored against the old")
        print(f"  solid. Delete those lines from results.jsonl and re-run them.")
    return 1 if (mismatched or unbuildable or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
