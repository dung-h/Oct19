#!/usr/bin/env python3
"""Continue the fixed workflow after an already-running extraction finishes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int);args=parser.parse_args()
    if args.wait_pid:
        while True:
            path=Path(f'/proc/{args.wait_pid}/stat')
            if not path.exists() or path.read_text().split()[2]=='Z':break
            time.sleep(5)
    source=HERE/'outputs/extraction.jsonl'
    records=[json.loads(line) for line in source.read_text().splitlines()]
    assert len(records)==3090 and len({r['key'] for r in records})==3090,'Extraction incomplete; do not train.'
    for command in [['roles.py','encode'],['test_roles.py'],['run.py','all'],['verify.py'],['report.py']]:
        print('START',command,flush=True)
        subprocess.run([sys.executable,'-u',str(HERE/command[0]),*command[1:]],check=True)
        print('COMPLETE',command,flush=True)


if __name__=='__main__':main()
