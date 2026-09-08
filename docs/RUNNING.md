# Running the benchmark

## 1. Environment

CadQuery no longer requires conda — the PyPI wheels work directly:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Headless machines need a software GL stack for the three-view render, or
`render.py` **segfaults** (it does not raise, so the crash takes the whole
benchmark process down on the first Judge call):

```bash
sudo apt-get install -y libosmesa6 libgl1 libglx-mesa0 libgl1-mesa-dri   # Debian/Ubuntu
```

macOS needs nothing extra.

Verify before starting a long run:

```bash
python -c "
from autofab.executor import Executor
from autofab.render import render_stl_to_png
r = Executor(output_dir='/tmp/chk').execute(
    'import cadquery as cq\nresult = cq.Workplane(\"XY\").box(50,30,20)', name='chk')
assert r.success, r.error
print('render:', render_stl_to_png(r.stl_path, '/tmp/chk/r.png'))
print('OK')"
```

## 2. Choose a backend

### Claude on Amazon Bedrock (`LLM_BACKEND=bedrock`)

Uses the ambient AWS credential chain — SSO profile, instance role, or env
vars. No Anthropic API key required.

```bash
aws sso login --profile my-profile

export LLM_BACKEND=bedrock
export AWS_PROFILE=my-profile
export AWS_REGION=us-east-1          # a region where Claude models are enabled
```

The Anthropic models must be enabled for your account in the Bedrock console
under **Model access** for the region you choose.

```bash
pip install "anthropic[bedrock]" boto3
```

### First-party Claude API (`LLM_BACKEND=anthropic`, the default)

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

## 3. Models

The pipeline pairs a coder with a **stronger judge** on purpose, so the Judge is
not grading its own homework. Defaults:

| Role | Default | Env override |
|---|---|---|
| Planner / Coder / Refiners | `claude-sonnet-5` | `CODER_MODEL` |
| Vision Judge | `claude-opus-5` | `JUDGE_MODEL` |

On Bedrock the `anthropic.` prefix is added automatically — set
`CODER_MODEL=claude-sonnet-5`, not the prefixed form.

### Output budget

| Role | Env | Default |
|---|---|---|
| Planner / Coder / Refiners | `CODER_MAX_TOKENS` | 16000 |
| Vision Judge | `MAX_TOKENS` | 16000 |

Do not lower either much. These models think adaptively by default, the
thinking is drawn from the same budget as the answer, and how long they think
grows with the difficulty of the request. Below roughly 8192 a hard T3 entry
can spend the whole budget thinking, so the reply arrives with **no text block
in it** and the runner records:

```
No text block in response (stop_reason=max_tokens)
```

That is a ceiling we set, not a model that could not build the part. A reply
cut off at the budget is retried once with the budget doubled, up to
`TRUNCATION_RETRY_CEILING` (32000), and the preflight warns about a budget
under the floor.

There is a ceiling at the other end too. Above `NONSTREAMING_MAX_TOKENS`
(16000, the documented non-streaming default) a request is refused unless it
streams, because a budget that large could take longer to generate than the
HTTP request may stay open:

```
Streaming is required for operations that may take longer than 10 minutes
```

The retry is larger than a budget that just proved too small, so it crosses
that line by design and is sent over a stream. `get_final_message()` returns
the same response object, so nothing downstream can tell which path was
taken. Raising `CODER_MAX_TOKENS` past 16000 is therefore safe - it streams.

To check all of this without spending anything:

```bash
python scripts/check_token_budget.py
```

If entries were already recorded with that error, drop them so a resumed run
redoes them:

```bash
python scripts/drop_failed_entries.py results/<experiment> --apply
python scripts/run_custom_benchmark.py --experiment-name <experiment> --tiers T1 T2 T3 --verbose
```

## 4. Run

```bash
# Full benchmark, all 100 entries
python scripts/run_custom_benchmark.py --experiment-name claude_full --verbose

# One tier
python scripts/run_custom_benchmark.py --experiment-name claude_t3 --tiers T3 --verbose

# A few entries first, to confirm wiring and sample the cost
python scripts/run_custom_benchmark.py --experiment-name smoke --ids T1_001 T2_001 --verbose

# Ablation: no vision for the Judge
python scripts/run_custom_benchmark.py --experiment-name claude_novision --no-vision
```

Results append to `results/<name>/results.jsonl`. The runner **skips entries
already recorded there**, so an interrupted run resumes by re-invoking the same
command. To force specific entries to re-run, pass them with `--ids`, or delete
their lines from `results.jsonl`.

## 5. Analyse

```bash
python scripts/analyze_results.py results/claude_full          # per-tier summary + CSV
python scripts/final_report.py  results/claude_full            # full report
python scripts/final_report.py  results/claude_full results/other_run   # head-to-head
```

### Read the metrics against the ceiling, not against 0 and 1

`data/dataset_v2/metric_ceiling.json` holds each entry's achievable score,
computed by comparing every reference STL **against itself** (ICP + 10k random
surface samples). A *perfect* answer does not score CD 0 / F1 1.0:

| Tier | CD floor (median) | F1 ceiling (median) |
|---|---|---|
| T1 | 0.3049 | 0.9985 |
| T2 | 0.3065 | 0.9987 |
| T3 | 0.9061 | 0.8881 |
| All | **0.4407** | **0.9899** |

Both floors are driven almost entirely by part size — surface area correlates
with the CD floor at **r = 1.000** and with the F1 ceiling at **r = −0.970**,
because mean nearest-neighbour spacing is `sqrt(area / 10000)` while F1's
threshold is fixed at τ = 1.0 mm. Large parts therefore cannot score well no
matter how correct they are: T1_031 is a *primitive* whose F1 ceiling is 0.583.

Two consequences when comparing runs:

1. Compare ceiling-relative numbers (`CD − CD_floor`, `F1 / F1_ceiling`), which
   `final_report.py` prints alongside the raw values.
2. Always quote **Exec%** and **n** next to CD/F1/IoU. A failed entry records
   `metrics: null` and is dropped before the medians, so a model whose worst
   cases crash outright can post a *better* median than one that produces
   bad-but-valid geometry.
