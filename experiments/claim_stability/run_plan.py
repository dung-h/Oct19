#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
for command in [
 [sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],
 [sys.executable,str(HERE/'run_claims.py'),'all'],
 [sys.executable,str(HERE/'verify_claims.py')],
 [sys.executable,str(HERE/'write_report.py')],
]:subprocess.run(command,check=True)
print('PLAN COMPLETE',flush=True)
