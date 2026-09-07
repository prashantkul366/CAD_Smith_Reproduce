"""Remove entries that failed for reasons outside the model's control.

The runner records every entry it attempts, including ones that failed because
credentials expired or the network dropped. Resume then skips them, so those
entries stay missing from the results and silently depress the tier totals.

This drops such records - and only such records - so the next run redoes them.
Genuine model or geometry failures are kept: they are results.

Usage:
    python scripts/drop_failed_entries.py results/<experiment>          # report
    python scripts/drop_failed_entries.py results/<experiment> --apply  # rewrite
"""

import argparse
import json
import shutil
from pathlib import Path

#: Substrings that mark a failure as infrastructure rather than model output.
INFRA_MARKERS = (
    "401",                      # expired or revoked credentials
    "ExpiredToken",
    "InvalidClientTokenId",
    "UnrecognizedClientException",
    "AuthenticationError",
    "credential",
    "Local LLM call",           # tunnel or local server unreachable
    "ConnectionError",
    "Connection error",
    "ReadTimeout",
    "Too Many Requests",
    "429",
    "ThrottlingException",
    "ServiceUnavailable",
    "503",
)


def is_infrastructure(record: dict) -> bool:
    error = str(record.get("error") or "")
    if not error:
        return False
    return any(marker.lower() in error.lower() for marker in INFRA_MARKERS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment_dir", type=Path)
    ap.add_argument("--apply", action="store_true",
                    help="rewrite results.jsonl (a .bak copy is kept)")
    args = ap.parse_args()

    path = args.experiment_dir / "results.jsonl"
    if not path.exists():
        print(f"ERROR: {path} not found")
        return 1

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    drop = [r for r in rows if is_infrastructure(r)]
    keep = [r for r in rows if not is_infrastructure(r)]

    print(f"{path}")
    print(f"  entries recorded          : {len(rows)}")
    print(f"  infrastructure failures   : {len(drop)}")
    print(f"  kept (results, incl. real failures): {len(keep)}")

    if drop:
        by_tier: dict[str, list[str]] = {}
        for r in drop:
            by_tier.setdefault(r.get("tier", "?"), []).append(r["id"])
        print("\n  to re-run:")
        for tier in sorted(by_tier):
            ids = by_tier[tier]
            shown = ", ".join(ids[:8]) + (" ..." if len(ids) > 8 else "")
            print(f"    {tier}: {len(ids):>3}  {shown}")
        print(f"\n  first error: {str(drop[0].get('error'))[:100]}")

    if not args.apply:
        print("\n  Dry run. Re-run with --apply to rewrite, then start the "
              "benchmark again with the same --experiment-name.")
        return 0

    if drop:
        backup = path.with_suffix(".jsonl.bak")
        shutil.copy(path, backup)
        with path.open("w", encoding="utf-8") as fh:
            for r in keep:
                fh.write(json.dumps(r) + "\n")
        print(f"\n  rewritten. backup at {backup}")
        print(f"  {len(drop)} entries will be redone on the next run.")
    else:
        print("\n  Nothing to drop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
