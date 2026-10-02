#!/usr/bin/env python3
"""Detach one bounded three-worker run and finish with an independent replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/natural_percentile/canonical_p_ablation_v1_20260925'
ARMS = ('condition_only','guidance_only','no_p')


def write_status(data):
    data['time_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = OUT / 'pipeline_status.tmp.json'
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(OUT / 'pipeline_status.json')


def monitor():
    freeze = json.loads((OUT/'freeze.json').read_text())
    assert freeze['status'] == 'pass'
    jobs, handles = {}, []
    for arm in ARMS:
        assert not (OUT/arm).exists(), 'No implicit restart or resampling'
        handle = (OUT/(arm+'.log')).open('x'); handles.append(handle)
        process = subprocess.Popen([sys.executable, str(ROOT/'pcontrol/research/p_path_ablation.py'), 'run', '--arm', arm],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2'))
        jobs[arm] = process
    prior = {}
    while True:
        states = {arm: dict(pid=p.pid, exit_code=p.poll(), completed_batches=len(list((OUT/arm).glob('case_*_z*.json')))//2,
                            result_exists=(OUT/arm/'result.json').exists()) for arm,p in jobs.items()}
        write_status(dict(stage='running', monitor_pid=os.getpid(), arms=states, expected_batches_per_arm=288))
        for arm,v in states.items():
            if v['result_exists'] and not prior.get(arm):
                print(json.dumps(dict(event='arm_completed', arm=arm, time_utc=datetime.now(timezone.utc).isoformat())), flush=True)
            prior[arm] = v['result_exists']
        if all(v['exit_code'] is not None for v in states.values()):
            break
        time.sleep(20)
    for h in handles:
        h.close()
    if any(v['exit_code'] != 0 or not v['result_exists'] for v in states.values()):
        write_status(dict(stage='failed_no_automatic_retry', monitor_pid=os.getpid(), arms=states))
        return
    write_status(dict(stage='independent_replay', monitor_pid=os.getpid(), arms=states))
    with (OUT/'audit.log').open('x') as handle:
        result = subprocess.run([sys.executable, str(ROOT/'pcontrol/research/audit_p_path_ablation.py')],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
    write_status(dict(stage='complete' if result.returncode==0 else 'audit_failed', monitor_pid=os.getpid(),
                      arms=states, audit_exit_code=result.returncode, comparison=str(OUT/'comparison.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--monitor', action='store_true'); a = p.parse_args()
    if a.monitor:
        monitor()
    else:
        assert not (OUT/'launch.json').exists(), 'Launch receipt already exists'
        log = (OUT/'pipeline.log').open('x')
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--monitor'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        log.close()
        launch = dict(monitor_pid=proc.pid, 
                      output=str(OUT), time_utc=datetime.now(timezone.utc).isoformat())
        with (OUT/'launch.json').open('x') as f: json.dump(launch,f,indent=2)
        print(json.dumps(launch),flush=True)
