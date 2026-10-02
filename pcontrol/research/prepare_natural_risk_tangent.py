#!/usr/bin/env python3
"""Prepare, compute one fixed shard, or aggregate the natural FIT tangent cache."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from pcontrol.generation.risk_tangent import prepare_cache,run_worker,aggregate_cache


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','worker','aggregate'),required=True)
    parser.add_argument('--policy');parser.add_argument('--policy-sha256')
    parser.add_argument('--prepared');parser.add_argument('--prepared-sha256');parser.add_argument('--shard',type=int)
    args=parser.parse_args()
    if args.stage=='prepare':
        if not args.policy or not args.policy_sha256:parser.error('prepare requires policy and SHA256')
        binding=prepare_cache(dict(path=args.policy,sha256=args.policy_sha256))
    else:
        if not args.prepared or not args.prepared_sha256:parser.error('worker/aggregate require prepared and SHA256')
        prepared=dict(path=args.prepared,sha256=args.prepared_sha256)
        if args.stage=='worker':
            if args.shard is None:parser.error('worker requires --shard0..7')
            binding=run_worker(prepared,args.shard)
        else:binding=aggregate_cache(prepared)
    print(json.dumps(dict(stage=args.stage,result=binding),indent=2),flush=True)


if __name__=='__main__':main()
