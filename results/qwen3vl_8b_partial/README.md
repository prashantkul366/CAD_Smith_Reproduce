# Qwen3-VL-8B-Instruct — partial run (58/100)

Local open-weights baseline run against this pipeline, served from a Colab
notebook through a Cloudflare quick tunnel. **Stopped at 58/100 by request**;
T1 is complete, T2 partial, T3 not started.

| Tier | Entries | Exec OK | Converged |
|---|---|---|---|
| T1 | 50/50 | 36 | 32 |
| T2 | 8/25 | 4 | 4 |
| T3 | 0/25 | — | — |

Config: full pipeline with vision, max 5 refinement iterations, 3 error
retries. The same 8B model served every role, including the Judge.

## Caveats that matter when comparing against a Claude run

**The Judge is not independent here.** Upstream CADSmith uses a stronger model
for the Judge than for the Coder specifically to avoid self-confirmation bias.
This run collapsed that — the same 8B model wrote and graded the code. On T1 it
approved 3 of 32 entries that were materially wrong (one at 14% of achievable
F1) and rejected 2 that were essentially perfect. So this run measures
generation *and* judging quality together.

**Not every failure is the model's.** T1 failures break down as: 6 code never
executed, 3 Planner emitted malformed JSON (no retry exists for the Planner,
unlike the Judge), 2 where refinement destroyed a working solid, 2 rejected by
the executor for returning a valid `cq.Solid` instead of a `cq.Workplane`, and
1 lost to a tunnel timeout. The last three groups are harness and
infrastructure, not capability — T1_004's rejected cone was geometrically
perfect (F1 = 1.000 against the reference).

**Entries recorded as transport failures were removed and re-run**, so no
Cloudflare loss remains in this file.

**Generated with a 4096-token budget**, non-streamed for the first 54 entries
and streamed thereafter. Streaming changed only how bytes were transported, not
what the model produced (greedy decoding, identical prompts).

Read the numbers against `data/dataset_v2/metric_ceiling.json` — see
`docs/RUNNING.md` for why raw CD/F1 are size-confounded.
