#!/usr/bin/env python3
"""One-shot process orchestration ONLY; scientific work uses frozen entrypoints.

No retries, training, reselection, population changes, or source edits. Individual
logs and append-only events survive interrupted observation of this supervisor.
"""
import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.research.run_final_natural_validation import authorized_context
from pcontrol.publication_pipeline.final_validation_protocol import verified_json, digest

POLICY = {"path": str(ROOT / "configs/natural_percentile/final_natural_validation_v1.json"),
          "sha256": "1b74d7eca578e6cb8d8693b87e693208aa2bbb78b6ce44f540cfaf58cb96e280"}
PREP = ROOT / "outputs/natural_percentile/transformer_publication_v1_20260916/final_validation_v1_preparation"
FREEZE = {"path": str(PREP / "execution_freeze_v1.json"),
          "sha256": "a39add6f3b5e7bfecea062d293356459f13960b64b9d5bc23dbc5d2b8c5ae346"}
APPROVAL = {"path": str(PREP / "unsealing_approval_20260916.json"),
            "sha256": "c53d5ca1f9b12410f3b76ef0f60b02f7dbb270d1ead83ae18af9ef3d7f7d5e45"}
SCOPES = ("TEST", "VAL", "R18")
LOCK = threading.Lock()


def main():
    contexts = {s: authorized_context(POLICY, FREEZE, APPROVAL, s) for s in SCOPES}
    policy = verified_json(POLICY)
    out = ROOT / policy["evaluation_root"]
    supervisor = out / "orchestration_v1"
    supervisor.mkdir(parents=True, exist_ok=False)
    events = supervisor / "events.jsonl"
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    common = []
    for name, binding in (("policy", POLICY), ("freeze", FREEZE), ("approval", APPROVAL)):
        common += ["--" + name, binding["path"], "--" + name + "-sha256", binding["sha256"]]

    def emit(event, **fields):
        row = dict(time_utc=datetime.now(timezone.utc).isoformat(), event=event, **fields)
        line = json.dumps(row, ensure_ascii=False)
        with LOCK:
            with events.open("a") as handle:
                handle.write(line + "\n")
            print(line, flush=True)

    def run(command, scope, *, recording=None, arm=None):
        key = "_".join(x for x in (scope, command, recording, arm) if x)
        cmd = [sys.executable, "-u", str(ROOT / "pcontrol/research/run_final_natural_validation.py"),
               command, *common, "--scope", scope]
        if recording is not None:
            cmd += ["--recording", recording]
        if arm is not None:
            cmd += ["--arm", arm]
        log = supervisor / (key + ".log")
        started = time.monotonic()
        with log.open("x") as handle:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            emit("stage_started", key=key, pid=process.pid, log=str(log), command=cmd)
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    emit("stage_running", key=key, pid=process.pid,
                         elapsed_seconds=round(time.monotonic() - started, 1), log_bytes=log.stat().st_size)
        emit("stage_completed" if code == 0 else "stage_failed", key=key, exit_code=code,
             elapsed_seconds=round(time.monotonic() - started, 1), log=str(log))
        if code:
            raise RuntimeError(f"Frozen stage failed: {key}; no automatic retry; inspect {log}")

    def bounded_jobs(jobs, workers):
        # Only already-running peers finish after failure; queued jobs are not started.
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            remaining = iter(jobs)
            active = {pool.submit(run, **job) for job in [next(remaining, None) for _ in range(workers)] if job}
            while active:
                done, active = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    future.result()
                for _ in done:
                    job = next(remaining, None)
                    if job is not None:
                        active.add(pool.submit(run, **job))

    emit("supervisor_started", pid=os.getpid(), script_sha256=digest(__file__),
         policy=POLICY, freeze=FREEZE, approval=APPROVAL,
         order=list(SCOPES), scientific_changes=False)
    try:
        for scope in SCOPES:
            bounded_jobs([dict(command="build-recording", scope=scope, recording=r)
                          for r in contexts[scope][0].recording_ids],
                         workers=policy["runtime"]["raw_recording_workers"])
            run("finalize-data", scope)
            run("reference", scope)
            run("queue", scope)
            emit("cohort_data_and_reference_complete", scope=scope)
        bounded_jobs([dict(command="generate", scope=s, arm=a)
                      for s in SCOPES for a in policy["generation"]["arms"]], workers=2)
        for scope in SCOPES:
            run("audit", scope)
            emit("cohort_complete", scope=scope)
    except BaseException as exc:
        emit("supervisor_failed", error=repr(exc), no_automatic_retry=True)
        raise
    emit("supervisor_complete", all_three_cohorts_complete=True, manuscript_update_pending=True)


if __name__ == "__main__":
    main()
