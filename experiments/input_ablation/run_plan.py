#!/usr/bin/env python3
"""Resume/run the authorized staged plan and verify all resulting artifacts."""
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from finalize_workflow import stage_decision


def run(path,*args):
    subprocess.run([sys.executable,str(path),*args],cwd=ROOT,check=True)


def main():
    run(HERE/'run_ablation.py','all')
    run(HERE/'verify_and_summarize.py')
    run(HERE/'audit_metadata.py')
    run(HERE/'run_metadata.py')
    run(HERE/'verify_metadata.py')
    decision=stage_decision()
    if decision['advance_to_adapter']:
        run(HERE.parent/'preference_adapter/run_adapter.py')
        run(HERE.parent/'preference_adapter/verify_adapter.py')
    run(HERE/'finalize_workflow.py')
    run(HERE/'verify_final.py')
    run(HERE/'write_report.py')


if __name__=='__main__':main()
