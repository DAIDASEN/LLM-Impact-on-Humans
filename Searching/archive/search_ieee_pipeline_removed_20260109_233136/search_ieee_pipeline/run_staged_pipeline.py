#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> bool:
    print("\n>>", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"Command failed with exit code {res.returncode}")
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage1 -> Stage2 -> Stage3 sequentially for a CSV of papers.")
    ap.add_argument("--csv", default="papers_for_pipeline.csv", help="Input CSV with papers (default: papers_for_pipeline.csv)")
    ap.add_argument("--papers_dir", default="papers", help="Directory where PDFs are stored (default: papers)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of rows (0 = no limit)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip_stage1", action="store_true")
    ap.add_argument("--skip_stage2", action="store_true")
    ap.add_argument("--skip_stage3", action="store_true")
    ap.add_argument("--require_stage1_pass_for_stage3", action="store_true", help="Pass --require_step1_pass to Stage3")
    args = ap.parse_args()

    csv_in = Path(args.csv)
    if not csv_in.exists():
        print(f"Input CSV not found: {csv_in}")
        sys.exit(1)

    # Derived paths
    stage1_out = Path("llm_outputs/validate.jsonl")
    stage2_out = Path("llm_outputs/screen_cluster.jsonl")
    stage3_out = Path("llm_outputs/extract_summary.jsonl")

    stage1_csv = csv_in.with_name("papers_stage1.csv")
    stage2_csv = csv_in.with_name("papers_stage2.csv")

    py = sys.executable

    # Stage 1
    if not args.skip_stage1:
        cmd1 = [py, "validate_paper.py", "--csv", str(csv_in), "--papers_dir", args.papers_dir, "--out", str(stage1_out), "--csv_out", str(stage1_csv)]
        if args.resume:
            cmd1.append("--resume")
        if args.limit and args.limit > 0:
            cmd1 += ["--limit", str(args.limit)]
        if args.start and args.start > 0:
            cmd1 += ["--start", str(args.start)]

        ok = run_cmd(cmd1)
        if not ok:
            print("Stage1 failed — aborting sequence.")
            sys.exit(1)
    else:
        print("Skipping Stage1 as requested")

    # Stage 2
    if not args.skip_stage2:
        # screen_cluster supports --only_stage1_match when CSV has stage1 columns
        cmd2 = [py, "screen_cluster.py", "--csv", str(stage1_csv if stage1_csv.exists() else csv_in), "--papers_dir", args.papers_dir, "--out", str(stage2_out), "--only_stage1_match", "--csv_out", str(stage2_csv)]
        if args.resume:
            cmd2.append("--resume")
        if args.limit and args.limit > 0:
            cmd2 += ["--limit", str(args.limit)]
        if args.start and args.start > 0:
            cmd2 += ["--start", str(args.start)]

        ok = run_cmd(cmd2)
        if not ok:
            print("Stage2 failed — aborting sequence.")
            sys.exit(1)
    else:
        print("Skipping Stage2 as requested")

    # Stage 3
    if not args.skip_stage3:
        cmd3 = [py, "extract_summary.py", "--csv", str(stage2_csv if stage2_csv.exists() else csv_in), "--papers_dir", args.papers_dir, "--stage1_jsonl", str(stage1_out), "--screen_jsonl", str(stage2_out), "--out", str(stage3_out)]
        if args.resume:
            cmd3.append("--resume")
        if args.limit and args.limit > 0:
            cmd3 += ["--limit", str(args.limit)]
        if args.start and args.start > 0:
            cmd3 += ["--start", str(args.start)]
        if args.require_stage1_pass_for_stage3:
            cmd3.append("--require_step1_pass")

        ok = run_cmd(cmd3)
        if not ok:
            print("Stage3 failed — sequence finished with errors.")
            sys.exit(1)
    else:
        print("Skipping Stage3 as requested")

    print("All requested stages completed successfully (or skipped).")


if __name__ == "__main__":
    main()
