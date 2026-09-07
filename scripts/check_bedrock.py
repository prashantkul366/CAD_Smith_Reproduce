"""Preflight for LLM_BACKEND=bedrock.

Checks, in order, the things that actually break a run:
  1. credentials resolve
  2. Bedrock is reachable and Anthropic models are enabled for this account/region
  3. the configured model IDs exist
  4. a real text call works through the repo's own agent code
  5. a real VISION call works (the Judge path, used every iteration)

Run:  python scripts/check_bedrock.py
"""
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OK, BAD, WARN = "  [OK]  ", "  [FAIL]", "  [WARN]"


def main() -> int:
    os.environ.setdefault("LLM_BACKEND", "bedrock")
    from autofab import agents

    print("=" * 68)
    print("  BEDROCK PREFLIGHT")
    print("=" * 68)
    print(f"  backend     : {agents.LLM_BACKEND}")
    print(f"  region      : {agents.AWS_REGION}")
    print(f"  profile     : {agents.AWS_PROFILE or '(none - using env credentials)'}")
    raw_profile = (os.getenv("AWS_PROFILE") or "").strip()
    if raw_profile and agents.AWS_PROFILE is None:
        print(f"{WARN} AWS_PROFILE is set to the placeholder '{raw_profile}' and is")
        print("         being ignored. Clear it with:")
        print("           PowerShell:  Remove-Item Env:AWS_PROFILE")
        print("           bash:        unset AWS_PROFILE")
    print(f"  coder model : {agents.CODER_MODEL}")
    print(f"  judge model : {agents.JUDGE_MODEL}")
    print(f"  max tokens  : {agents.MAX_TOKENS}")

    if agents.LLM_BACKEND != "bedrock":
        print(f"\n{BAD} LLM_BACKEND is '{agents.LLM_BACKEND}', not 'bedrock'.")
        print("        PowerShell:  $env:LLM_BACKEND='bedrock'")
        return 1

    # --- 1. credentials -----------------------------------------------------
    print("\n[1/5] credentials")
    try:
        import boto3
        ident = boto3.client("sts", region_name=agents.AWS_REGION).get_caller_identity()
        print(f"{OK} account {ident['Account']}")
        print(f"       {ident['Arn']}")
    except Exception as e:
        print(f"{BAD} {type(e).__name__}: {e}")
        print("        Credentials missing or expired. Re-copy them from")
        print("        https://ymsli.awsapps.com/start -> your role -> Access keys")
        return 1

    # --- 2. model access ----------------------------------------------------
    print("\n[2/5] Anthropic models enabled in this region")
    available = []
    try:
        bl = boto3.client("bedrock", region_name=agents.AWS_REGION)
        for m in bl.list_foundation_models().get("modelSummaries", []):
            if "anthropic" in m["modelId"].lower():
                available.append(m["modelId"])
        if available:
            print(f"{OK} {len(available)} Anthropic model(s) visible:")
            for m in sorted(available):
                print(f"         {m}")
        else:
            print(f"{WARN} none visible. Enable them in the Bedrock console under")
            print("        'Model access' for this region, or try another region.")
    except Exception as e:
        print(f"{WARN} could not list models ({type(e).__name__}: {e})")
        print("        Not fatal - the role may lack bedrock:ListFoundationModels")
        print("        but still be able to invoke. Continuing.")

    # --- 3. configured IDs --------------------------------------------------
    print("\n[3/5] configured model IDs")
    for label, mid in (("coder", agents.CODER_MODEL), ("judge", agents.JUDGE_MODEL)):
        bare = mid.split("anthropic.", 1)[-1]
        hit = [a for a in available if bare in a or a in mid]
        if not available:
            print(f"{WARN} {label}: {mid} (cannot verify - listing unavailable)")
        elif hit:
            print(f"{OK} {label}: {mid}")
        else:
            print(f"{WARN} {label}: {mid} not found in the list above.")
            print(f"         Override with:  $env:{label.upper()}_MODEL='<id>'")

    # --- 4. text call -------------------------------------------------------
    print("\n[4/5] live text call (coder model)")
    try:
        agents.reset_token_usage()
        out = agents._call_claude(
            "You are a CAD engineer. Output ONLY Python code, no prose, no fences.",
            "CadQuery for a 50x30x20mm block. Assign the result to `result`.",
        )
        print(f"{OK} responded, {len(out)} chars, usage={agents.get_token_usage()}")
        print("        " + out.strip().replace("\n", "\n        ")[:220])
    except Exception as e:
        print(f"{BAD} {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # --- 5. vision call -----------------------------------------------------
    # The Judge runs on every iteration and is the only path that sends an
    # image, so a failure here breaks the whole loop, not just one entry.
    print("\n[5/5] live VISION call (judge model + rendered image)")
    try:
        from autofab.executor import Executor
        import tempfile
        tmp = tempfile.mkdtemp(prefix="bedrock_check_")
        r = Executor(output_dir=tmp, timeout_seconds=120).execute(
            "import cadquery as cq\nresult = cq.Workplane('XY').box(50,30,20)\n", name="chk")
        if not r.success:
            print(f"{BAD} CadQuery execution failed: {r.error_type}")
            print("        This is a local CAD/env problem, not a Bedrock one.")
            return 1
        agents.reset_token_usage()
        verdict = agents.evaluate_geometry(
            prompt="A rectangular block 50mm long, 30mm wide, and 20mm tall.",
            code="import cadquery as cq\nresult = cq.Workplane('XY').box(50,30,20)",
            geometry_metrics=r.geometry_json,
            stl_path=r.stl_path,
            render_save_path=os.path.join(tmp, "render.png"),
        )
        print(f"{OK} judge returned passed={verdict.get('passed')}, "
              f"usage={agents.get_token_usage()}")
        print(f"        {str(verdict.get('feedback'))[:200]}")
        if agents.get_token_usage()["input_tokens"] < 500:
            print(f"{WARN} input tokens look low - the render may not have been")
            print("         attached. Check VTK is installed and working.")
    except Exception as e:
        print(f"{BAD} {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    print("\n" + "=" * 68)
    print("  ALL CHECKS PASSED - ready to run the benchmark")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
