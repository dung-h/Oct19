#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
commands=[
 [sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],
 [sys.executable,str(HERE/'run_diagnosis.py'),'all'],
 [sys.executable,str(HERE/'materialize_reviews.py')],
 [sys.executable,str(HERE/'verify_diagnosis.py')],
 [sys.executable,str(HERE/'write_summary.py')],
]
for cmd in commands:subprocess.run(cmd,check=True)
print('PLAN COMPLETE',flush=True)
