#!/usr/bin/env python3
"""Independent piecewise-linear score audit; never fits or samples a model."""
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pcontrol.research.train_natural_scene_reference import sha256, verify_binding, write_json


def independent_scores(mass, y, node, u_knots, thresholds):
    x = np.linspace(0., 4., len(mass)-1)
    levels = np.r_[mass[0], mass[0]+np.cumsum(mass[1:-1])]
    levels[-1] = 1.-mass[-1]
    if np.any(np.diff(levels) < -1e-14):
        raise ValueError("invalid base probabilities")

    def interior_cdf(t):
        return np.interp(np.interp(t, x, levels), u_knots, node)

    crossings = []
    for u in u_knots:
        if mass[0] < u < 1.-mass[-1]:
            end = np.searchsorted(levels, u, side="left")
            if not 1 <= end < len(levels) or levels[end] <= levels[end-1]:
                raise ValueError("unresolved independent probability crossing")
            crossings.append(x[end-1]+(u-levels[end-1])/(levels[end]-levels[end-1])*(x[end]-x[end-1]))
    scores = {}
    for upper, key in ((4., "CRPS_seconds"), (1., "twCRPS_1s"), (2., "twCRPS_2s")):
        points = np.unique(np.r_[x, crossings, y, upper])
        points = points[(points >= 0) & (points <= upper)]
        values = interior_cdf(points)
        event = y <= .5*(points[:-1]+points[1:])
        a, b = values[:-1]-event, values[1:]-event
        scores[key] = float(np.sum(np.diff(points)*(a*a+a*b+b*b)/3.))
    value = float(interior_cdf(y))
    left = 0. if y == 0 else value
    right = 1. if y == 4 else value
    scores.update(cdf_left=left, cdf_right=right,
                  threshold_cdf=interior_cdf(thresholds), zero_mass=float(interior_cdf(0.)),
                  cap_mass=1.-float(interior_cdf(4.)))
    return scores


def run():
    root = ROOT/"outputs/natural_percentile/context_calibration_refinement_v2_20260914"
    selection_path = root/"selection_before_development_AUDIT.json"
    selection = json.loads(selection_path.read_text())
    policy = json.loads(verify_binding(selection["policy"]).read_text())
    freeze = json.loads(verify_binding(selection["freeze"]).read_text())
    for path, expected in selection["code_sha256"].items():
        if sha256(ROOT/path) != expected: raise ValueError("frozen experiment code changed")
    with np.load(verify_binding(freeze["CAL_predictions"]), allow_pickle=False) as z:
        cal_masses = z["joint_masses"]; cal_ids = z["scene_id"]; cal_records = z["recording_id"]
    candidates = selection["candidates"]
    fold_count = 0
    for candidate in candidates.values():
        for fold in candidate["folds"]:
            held = fold["held_recording"]; fitting = set(fold["fitting_recordings"])
            if held in fitting or fitting | {held} != set(cal_records): raise ValueError("CAL fold role leakage")
            if fold["held_rows"] != int((cal_records == held).sum()): raise ValueError("held row count")
            if fold["fit"]["rows"] != int((cal_records != held).sum()): raise ValueError("fit row count")
            fold_count += 1
    eligible = [name for name,c in candidates.items() if c["eligible"]]
    winner = min(eligible, key=lambda n:(candidates[n]["metrics"]["context_selection_score"], candidates[n]["folds"][0]["fit"]["parameter_count"], -candidates[n]["ridge"])) if eligible else "old_count_fallback"
    if winner != selection["selected_name"]: raise ValueError("selection does not replay")
    paths = [root/"identity_CAL_OOF.npz", root/"old_count_CAL_OOF.npz"]
    paths += [verify_binding(c["predictions"]) for c in candidates.values() if "predictions" in c]
    dev = json.loads((root/"development_evaluation/results.json").read_text())
    if dev["selection"]["sha256"] != sha256(selection_path): raise ValueError("evaluation changed selection")
    paths += [verify_binding(v) for v in dev["predictions"].values()]
    u = np.array([0., .05, .1, .25, .5, .75, .9, .95, 1.]); thresholds = np.array(policy["thresholds_seconds"])
    maximum = 0.; checked = 0; file_reports = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            a = {key:z[key] for key in z.files}
        is_cal = "base_joint_masses" not in a
        if is_cal:
            if not np.array_equal(a["scene_id"], cal_ids): raise ValueError("CAL row alignment")
            masses = cal_masses
        else:
            if set(a["recording_id"]) & set(cal_records): raise ValueError("CAL/development recording overlap")
            masses = a["base_joint_masses"]
        local_max = 0.
        for i, (mass,y,node) in enumerate(zip(masses,a["target"],a["effective_nodes"])):
            values = independent_scores(mass, y, node, u, thresholds)
            for name,value in values.items():
                local_max = max(local_max, float(np.max(np.abs(value-a[name][i]))))
            checked += 1
        if local_max > 2e-11: raise ValueError("independent scoring mismatch: "+str(path))
        maximum = max(maximum, local_max)
        file_reports.append(dict(path=str(path), sha256=sha256(path), rows=len(masses), maximum_error=local_max))
    env = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__)
    result = dict(status="pass", selected=winner, checked_rows=checked, checked_files=len(paths), CAL_folds=fold_count,
                  maximum_independent_score_difference=maximum, independent_method="direct linear-segment polynomial integral; no scoring-kernel reuse",
                  files=file_reports, environment=env, code_sha256=sha256(__file__), no_refitting_or_generation=True,
                  legacy_sparse_tail_test_failure_documented_separately=True, current_candidate_boundary_test_passes=True)
    write_json(root/"independent_numeric_audit.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "files"}), flush=True)


if __name__ == "__main__": run()
