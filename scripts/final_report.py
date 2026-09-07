"""Full CADSmith benchmark report: as-written vs harness-normalized,
with ceiling-relative scoring and a model/harness/infrastructure failure split.

Usage: python3 final_report.py <run1_results_dir> [run2_results_dir]
"""
import json, sys, os, pathlib
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_CEIL_PATH = _HERE.parent / "data" / "dataset_v2" / "metric_ceiling.json"
CEIL = json.load(open(_CEIL_PATH)) if _CEIL_PATH.exists() else {}
TIERS = ('T1', 'T2', 'T3')


def load(d):
    p = os.path.join(d, 'results.jsonl')
    if not os.path.exists(p):
        return []
    out = []
    for l in open(p):
        try: out.append(json.loads(l))
        except json.JSONDecodeError: pass
    return out


def classify(r):
    """Attribute an unscored entry to model / harness / infrastructure."""
    if r.get('metrics'):
        return 'OK'
    e = str(r.get('error') or '')
    if 'Local LLM call' in e:
        return 'INFRA_524'
    # Planner died before any iteration -> JSON parse error out of _extract_json
    if r.get('num_iterations') in (None, 0) and any(
            k in e for k in ('Expecting', 'Extra data', 'delimiter', 'Invalid', 'JSON')):
        return 'MODEL_PLANNER_JSON'
    its = r.get('per_iteration') or []
    types = {rt.get('error_type') for i in its for rt in (i.get('error_retries') or [])}
    if 'NoResultError' in types:
        return 'HARNESS_NON_WORKPLANE'
    if any(i.get('execution_success') for i in its):
        return 'MODEL_HARNESS_REGRESSION'   # had a valid solid, refinement destroyed it
    if its:
        return 'MODEL_EXEC_FAILED'
    return 'OTHER'


def best_available(r):
    """Metrics of the last successfully-executed iteration.

    The runner reports metrics only for the FINAL iteration, so an entry whose
    refinement broke a previously-valid solid contributes nothing to any median.
    This recovers the best solid the pipeline actually produced.
    """
    if r.get('metrics'):
        return r['metrics'], False
    for it in reversed(r.get('per_iteration') or []):
        if it.get('metrics'):
            return it['metrics'], True
    return None, False


def agg(vals):
    return (np.median(vals), np.mean(vals)) if vals else (float('nan'),) * 2


def tier_block(rows, label):
    print(f"\n{'='*78}\n  {label}\n{'='*78}")
    hdr = (f"  {'Tier':<5}{'n':>4}{'Exec%':>8}{'Conv%':>8}{'Scored':>8}"
           f"{'CD med':>9}{'CD mean':>10}{'F1 med':>9}{'IoU med':>9}")
    print(hdr); print('  ' + '-' * (len(hdr) - 2))
    for t in TIERS + ('ALL',):
        d = rows if t == 'ALL' else [r for r in rows if r.get('tier') == t]
        if not d: continue
        sc = [r for r in d if r.get('metrics')]
        cd = [r['metrics']['chamfer_distance'] for r in sc]
        f1 = [r['metrics']['f1_score'] for r in sc]
        io = [r['metrics']['volumetric_iou'] for r in sc]
        cm, cu = agg(cd); fm, _ = agg(f1); im, _ = agg(io)
        print(f"  {t:<5}{len(d):>4}{sum(1 for r in d if r.get('execution_success'))/len(d)*100:>7.1f}%"
              f"{sum(1 for r in d if r.get('converged'))/len(d)*100:>7.1f}%{len(sc):>8}"
              f"{cm:>9.4f}{cu:>10.4f}{fm:>9.4f}{im:>9.4f}")


def ceiling_block(rows, label):
    print(f"\n  --- ceiling-relative ({label}) ---")
    print(f"  {'Tier':<5}{'n':>4}{'CDfloor':>10}{'CDexcess med':>14}{'CDexcess mean':>15}"
          f"{'F1/ceil med':>13}{'% at ceiling':>14}")
    for t in TIERS + ('ALL',):
        d = rows if t == 'ALL' else [r for r in rows if r.get('tier') == t]
        sc = [r for r in d if r.get('metrics') and r['id'] in CEIL]
        if not sc: continue
        fl = [CEIL[r['id']]['chamfer_distance'] for r in sc]
        ex = [r['metrics']['chamfer_distance'] - CEIL[r['id']]['chamfer_distance'] for r in sc]
        ra = [r['metrics']['f1_score'] / CEIL[r['id']]['f1_score'] for r in sc]
        at = sum(1 for x in ra if x > 0.98)
        print(f"  {t:<5}{len(sc):>4}{np.median(fl):>10.4f}{np.median(ex):>+14.4f}"
              f"{np.mean(ex):>+15.4f}{np.median(ra):>13.4f}{at/len(sc)*100:>13.0f}%")


def recovery_block(rows, label):
    print(f"\n  --- best-available-solid ({label}: last successful iteration counted) ---")
    print(f"  {'Tier':<5}{'scored':>8}{'recovered':>11}{'CD med':>9}{'CD mean':>10}{'F1 med':>9}")
    for t in TIERS + ('ALL',):
        d = rows if t == 'ALL' else [r for r in rows if r.get('tier') == t]
        if not d: continue
        got, rec = [], 0
        for r in d:
            m, was = best_available(r)
            if m: got.append(m); rec += was
        if not got: continue
        cd = [m['chamfer_distance'] for m in got]; f1 = [m['f1_score'] for m in got]
        print(f"  {t:<5}{len(got):>8}{rec:>11}{np.median(cd):>9.4f}{np.mean(cd):>10.4f}{np.median(f1):>9.4f}")


def failure_block(rows, label):
    print(f"\n  --- failure attribution ({label}) ---")
    cats = {}
    for r in rows:
        cats.setdefault(classify(r), []).append(r['id'])
    order = ['OK', 'MODEL_EXEC_FAILED', 'MODEL_PLANNER_JSON', 'MODEL_HARNESS_REGRESSION',
             'HARNESS_NON_WORKPLANE', 'INFRA_524', 'OTHER']
    tot = len(rows)
    for k in order:
        if k not in cats: continue
        v = cats[k]
        print(f"  {k:<26}{len(v):>4} ({len(v)/tot*100:>4.0f}%)  {', '.join(v[:9])}{' ...' if len(v)>9 else ''}")
    model = sum(len(cats.get(k, [])) for k in
                ('MODEL_EXEC_FAILED', 'MODEL_PLANNER_JSON', 'MODEL_HARNESS_REGRESSION'))
    harness = len(cats.get('HARNESS_NON_WORKPLANE', []))
    infra = len(cats.get('INFRA_524', []))
    print(f"\n  => model {model} | harness {harness} | infrastructure {infra}"
          f" | of {tot - len(cats.get('OK', []))} total failures")
    return cats


def judge_block(rows, label):
    fa = fr = ok = 0; det = []
    for r in rows:
        if not r.get('metrics') or r['id'] not in CEIL: continue
        rel = r['metrics']['f1_score'] / CEIL[r['id']]['f1_score']
        if r.get('converged') and rel < 0.90:
            fa += 1; det.append((r['id'], rel))
        elif r.get('converged'): ok += 1
        elif rel > 0.98: fr += 1
    n = ok + fa
    print(f"\n  --- judge reliability ({label}) ---")
    if n:
        print(f"  approved {n}, of which materially wrong (false accept): {fa} ({fa/n*100:.0f}%)")
        for i, rel in sorted(det, key=lambda x: x[1])[:6]:
            print(f"     {i}: {rel*100:.0f}% of achievable F1")
    print(f"  rejected an essentially perfect answer (false reject): {fr}")


def instrumentation(rows, label):
    rep = {}
    for r in rows:
        for k, v in (r.get('repairs') or {}).items():
            rep[k] = rep.get(k, 0) + v
    noop = sum(r.get('refiner_noop_total', 0) for r in rows)
    ti = sum((r.get('tokens') or {}).get('input_tokens', 0) for r in rows)
    to = sum((r.get('tokens') or {}).get('output_tokens', 0) for r in rows)
    print(f"\n  --- instrumentation ({label}) ---")
    c = rep.get('calls', 0)
    if c:
        print(f"  code extractions: {c}")
        for lab, k in (("missing `result =`", 'result_patched'),
                       ("missing cadquery import", 'import_patched'),
                       ("prose before code", 'prose_stripped')):
            print(f"    {lab:<26}{rep.get(k,0):>5} ({rep.get(k,0)/c*100:>4.1f}%)")
    print(f"  Error-Refiner no-op returns: {noop}")
    print(f"  tokens: {ti:,} in / {to:,} out")


def report(d, label):
    rows = load(d)
    if not rows:
        print(f"\n!! no results at {d}"); return None
    print(f"\n\n{'#'*78}\n#  {label}  —  {len(rows)}/100 entries\n{'#'*78}")
    tier_block(rows, f"{label}: AS MEASURED")
    ceiling_block(rows, label)
    recovery_block(rows, label)
    failure_block(rows, label)
    judge_block(rows, label)
    instrumentation(rows, label)
    return rows


if __name__ == '__main__':
    a = report(sys.argv[1], 'RUN 1 — AS-WRITTEN HARNESS')
    b = report(sys.argv[2], 'RUN 2 — HARNESS-NORMALIZED') if len(sys.argv) > 2 else None
    if a and b:
        print(f"\n\n{'#'*78}\n#  HEAD-TO-HEAD\n{'#'*78}")
        ai = {r['id']: r for r in a}; bi = {r['id']: r for r in b}
        common = sorted(set(ai) & set(bi))
        print(f"  common entries: {len(common)}")
        rescued = [i for i in common if not ai[i].get('metrics') and bi[i].get('metrics')]
        lost    = [i for i in common if ai[i].get('metrics') and not bi[i].get('metrics')]
        print(f"  rescued by normalization: {len(rescued)} {rescued}")
        print(f"  regressed:                {len(lost)} {lost}")
        for lbl, rows_, idx in (('run1', a, ai), ('run2', b, bi)):
            sc = [idx[i] for i in common if idx[i].get('metrics')]
            cd = [r['metrics']['chamfer_distance'] for r in sc]
            print(f"  {lbl}: exec {sum(1 for i in common if idx[i].get('execution_success'))}/{len(common)}"
                  f" | scored {len(sc)} | CD med {np.median(cd):.4f} mean {np.mean(cd):.4f}")
