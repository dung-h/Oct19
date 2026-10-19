#!/usr/bin/env python3
"""Run the abstract-preference margin experiment."""
import argparse

from apm_runner import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "outer", "final", "all"], nargs="?", default="all")
    main(parser.parse_args().stage)
