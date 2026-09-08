"""Check the output-token budget the text agents run with, offline.

The failure this guards against cost eight entries of a 100-entry benchmark
run - seven of them T3, the hardest tier - and none of them was a model that
could not build the part. Thinking tokens are drawn from the same budget as
the answer, and these models think for longer the harder the request is. At a
4096-token ceiling the hardest entries spent the whole budget thinking, so the
reply came back with no text block in it and the runner recorded:

    No text block in response (stop_reason=max_tokens)

Every check here runs against a fake client, so it needs no credentials, no
network and no money. Run it before a long benchmark:

    python scripts/check_token_budget.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from autofab import agents  # noqa: E402

OK, BAD = "  ok  ", "  FAIL"
_failures = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"{OK if passed else BAD}  {name}" + (f"  [{detail}]" if detail else ""))
    if not passed:
        _failures.append(name)


# ---------------------------------------------------------------------------
# A model that thinks before it answers, and thinks longer on harder requests.
# ---------------------------------------------------------------------------

class _Block:
    def __init__(self, type_, text=None):
        self.type, self.text = type_, text


class _Usage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


class _Response:
    def __init__(self, blocks, stop_reason, used):
        self.content, self.stop_reason = blocks, stop_reason
        self.usage = _Usage(437, used)


class FakeModel:
    """Spends `thinking_tokens` before writing anything.

    Under that budget the reply is thinking only and stops at max_tokens -
    exactly what Bedrock returned for the failed entries. At or above it the
    answer arrives.
    """

    def __init__(self, thinking_tokens: int):
        self.thinking_tokens = thinking_tokens
        self.budgets = []
        self.messages = self

    def create(self, *, max_tokens, **kwargs):
        self.budgets.append(max_tokens)
        if max_tokens < self.thinking_tokens:
            return _Response([_Block("thinking")], "max_tokens", max_tokens)
        return _Response([_Block("thinking"), _Block("text", "import cadquery as cq")],
                         "end_turn", self.thinking_tokens + 200)


def call(model: FakeModel, **kwargs):
    """Run _call_claude against a fake client, with usage counters reset."""
    agents.reset_token_usage()
    agents._transport_stats["budget_raises"] = 0
    original, agents._get_client = agents._get_client, lambda: model
    try:
        return agents._call_claude("system", "make me a bracket", **kwargs)
    finally:
        agents._get_client = original


def main() -> int:
    print("=" * 68)
    print("  OUTPUT TOKEN BUDGET")
    print("=" * 68)
    print(f"  text agents (CODER_MAX_TOKENS): {agents.CODER_MAX_TOKENS}")
    print(f"  vision judge (MAX_TOKENS)     : {agents.MAX_TOKENS}")
    print(f"  thinking floor                : {agents.THINKING_FLOOR}")
    print(f"  retry ceiling                 : {agents.TRUNCATION_RETRY_CEILING}\n")

    # 1. The configured budget clears the floor.
    check("the text agents' budget clears the thinking floor",
          agents.CODER_MAX_TOKENS >= agents.THINKING_FLOOR,
          f"{agents.CODER_MAX_TOKENS} >= {agents.THINKING_FLOOR}")
    check("the judge's budget clears it too",
          agents.MAX_TOKENS >= agents.THINKING_FLOOR,
          f"{agents.MAX_TOKENS} >= {agents.THINKING_FLOOR}")

    # 2. The reported failure, reproduced. The old code made one call and read
    #    the text off it, which is what these two lines do.
    hard = FakeModel(thinking_tokens=6000)
    reply = hard.messages.create(max_tokens=4096, model="x", system="s", messages=[])
    try:
        agents._response_text(reply)
        check("at 4096 a hard entry comes back with no text (the reported "
              "failure)", False, "no error raised")
    except ValueError as e:
        check("at 4096 a hard entry comes back with no text (the reported "
              "failure)",
              "No text block in response" in str(e) and "max_tokens" in str(e),
              str(e)[:58])

    # ... and the retry recovers that same entry, so a budget somebody has
    # pinned low in their .env costs a second call rather than the entry.
    hard = FakeModel(thinking_tokens=6000)
    out = call(hard, max_tokens=4096)
    check("the retry recovers it even from a budget that low", "cadquery" in out,
          f"budgets tried: {hard.budgets}")

    # 3. The same entry at the shipped default.
    hard = FakeModel(thinking_tokens=6000)
    out = call(hard)
    check("the shipped default answers it", "cadquery" in out,
          f"budgets tried: {hard.budgets}")
    check("and needed no retry", hard.budgets == [agents.CODER_MAX_TOKENS],
          str(hard.budgets))

    # 4. An entry that thinks past even that budget: one retry, then an answer.
    hardest = FakeModel(thinking_tokens=agents.CODER_MAX_TOKENS + 1000)
    out = call(hardest)
    check("a reply cut off at the budget is retried with more room",
          "cadquery" in out, f"budgets tried: {hardest.budgets}")
    check("the retry raised the budget", len(hardest.budgets) == 2
          and hardest.budgets[1] > hardest.budgets[0], str(hardest.budgets))
    check("the retry is counted as a transport event",
          agents.get_transport_stats()["budget_raises"] == 1,
          str(agents.get_transport_stats()["budget_raises"]))

    # 5. Retried once, not forever: a model that never answers must not loop.
    never = FakeModel(thinking_tokens=10 ** 9)
    try:
        call(never)
        check("a reply that never fits still raises", False, "no error raised")
    except ValueError as e:
        check("a reply that never fits still raises",
              "No text block in response" in str(e), str(e)[:50])
    check("and was retried exactly once", len(never.budgets) == 2,
          f"{len(never.budgets)} calls: {never.budgets}")
    check("the retry stayed under the ceiling",
          max(never.budgets) <= agents.TRUNCATION_RETRY_CEILING,
          str(never.budgets))

    # 6. A budget below the floor is called out rather than failing silently.
    agents._low_budget_warned.clear()
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            call(FakeModel(thinking_tokens=10 ** 9), max_tokens=2048)
        except ValueError:
            pass
    check("a budget under the floor is said out loud",
          "below the" in buf.getvalue() and "floor" in buf.getvalue(),
          buf.getvalue().strip().splitlines()[0][:60] if buf.getvalue() else "silent")

    # 7. `truncated` is per entry, not per run: the runner writes this dict
    #    into each record, so a counter left standing blames one entry for
    #    every truncation in the run.
    agents._token_usage["truncated"] = 7
    agents.reset_token_usage()
    check("reset_token_usage clears the truncation counter",
          "truncated" not in agents.get_token_usage(),
          str(agents.get_token_usage()))

    # 8. The entries already recorded can be reclaimed for a rerun.
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import drop_failed_entries as drop
    recorded = {"id": "T3_012", "tier": "T3", "success": False,
                "error": "No text block in response (stop_reason=max_tokens)"}
    check("a truncated entry is droppable, so a resume redoes it",
          drop.is_droppable(recorded), "")
    check("a real geometry failure is not",
          not drop.is_droppable({"id": "T3_001", "error": "IoU 0.31 below threshold"}),
          "")

    print()
    if _failures:
        print(f"  {len(_failures)} check(s) failed: {', '.join(_failures)}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
