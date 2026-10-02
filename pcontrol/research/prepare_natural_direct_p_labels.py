#!/usr/bin/env python3
"""Prepare, fit one fresh OOF teacher, or aggregate natural direct-P labels."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from pcontrol.reference.direct_p_crossfit import prepare,run_fold,aggregate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','fold','aggregate'),required=True)
    parser.add_argument('--policy');parser.add_argument('--policy-sha256')
    parser.add_argument('--prepared');parser.add_argument('--prepared-sha256')
    parser.add_argument('--fold',type=int)
    args=parser.parse_args()
    if args.stage=='prepare':
        if not args.policy or not args.policy_sha256:parser.error('prepare requires policy path and SHA256')
        result=prepare(dict(path=args.policy,sha256=args.policy_sha256))
    else:
        if not args.prepared or not args.prepared_sha256:parser.error('fold/aggregate require prepared path and SHA256')
        binding=dict(path=args.prepared,sha256=args.prepared_sha256)
        if args.stage=='fold':
            if args.fold is None:parser.error('fold stage requires --fold0..4')
            result=run_fold(binding,args.fold)
        else:result=aggregate(binding)
    print(json.dumps(dict(stage=args.stage,result=result),indent=2),flush=True)


if __name__=='__main__':main()
