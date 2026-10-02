#!/usr/bin/env python3
"""Prepare highD scenes for the reference and the generator.

For each recording, histories are taken on a 1 s grid of (ego, frame) candidates in a fixed hash order, as in the
paper: the ego and every same-direction vehicle within 120 longitudinal meters whose 13-state history (0.96 s) is
observed, at least three vehicles. A scene is kept when every one of its vehicles is observed over the whole
175-state future (6.96 s at 25 Hz). Its minimum ego--SV PET (capped at 4 s) is computed from vehicle footprints.

Usage:
    python pcontrol/tools/prepare_highd.py --raw-root /path/to/highD/data --recordings 35 47 --out data/scenes
    python pcontrol/tools/prepare_highd.py --raw-root /path/to/highD/data --split test --out data/scenes

Writes <out>/<recording>/complete_scenes.npz (read it with pcontrol.api.load_scenes) and a summary.json.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pcontrol.data.complete_scene_view import complete_eligibility, validate_complete_labels  # noqa: E402
from pcontrol.data.highd import (META_COLUMNS, TRACK_COLUMNS, HighDContractError, RawRecording,  # noqa: E402
                                 TrackMetadata, _lane_markings, recording_id)
from pcontrol.data.scene_pet import PROTOCOL as PET_PROTOCOL, scene_occupancy_pet  # noqa: E402
from pcontrol.data.scenes import extract_scene, scene_arrays, selection_cohort_rows  # noqa: E402
from pcontrol.publication_pipeline.final_scene_data import json_safe, select_history_cohort  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# Settings of the paper (configs/natural_percentile/final_natural_validation_v1.json, "data").
QUOTA, SALT = 4096, 'natural_ego_scene_v1'


def load_recording(raw_root, rec):
    """Read one highD recording (<rec>_tracks.csv, <rec>_tracksMeta.csv, <rec>_recordingMeta.csv)."""
    rec = recording_id(rec)
    paths = {key: Path(raw_root) / f'{rec}_{key}.csv' for key in ('tracks', 'tracksMeta', 'recordingMeta')}
    with paths['recordingMeta'].open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1 or recording_id(rows[0]['id']) != rec or float(rows[0]['frameRate']) != 25.:
        raise HighDContractError('one matching 25 Hz recording metadata row required')
    record = rows[0]
    tracks = pd.read_csv(paths['tracks'], usecols=list(TRACK_COLUMNS))
    table = pd.read_csv(paths['tracksMeta'], usecols=list(META_COLUMNS))
    if table['id'].duplicated().any():
        raise HighDContractError('duplicate tracksMeta actor')
    metadata = {}
    for row in table.to_dict('records'):
        ints = np.asarray([row[k] for k in ('id', 'initialFrame', 'finalFrame', 'drivingDirection')], dtype=np.float64)
        if not np.isfinite(ints).all() or not np.equal(ints, np.floor(ints)).all():
            raise HighDContractError('integer actor/frame/direction metadata required')
        actor, first, last, direction = map(int, ints)
        length, width = float(row['width']), float(row['height'])
        if actor <= 0 or first <= 0 or last < first or direction not in (1, 2):
            raise HighDContractError('invalid actor metadata')
        if not np.isfinite([length, width]).all() or min(length, width) <= 0:
            raise HighDContractError('invalid dimensions')
        metadata[actor] = TrackMetadata(first, last, length, width, str(row['class']), direction)
    return RawRecording(rec, tracks, metadata, 25., _lane_markings(record['upperLaneMarkings']),
                        _lane_markings(record['lowerLaneMarkings']), {k: str(v) for k, v in paths.items()}, 'highd_csv')


def complete_scenes(recording, selected, role):
    """Scenes whose vehicles are all observed over the future window, with their minimum ego--SV PET."""
    cohort = selection_cohort_rows(selected)
    complete, labels, kept, indices, counts = [], [], [], [], Counter()
    for index, item in enumerate(selected):
        context = item['context']
        scene, reason = extract_scene(recording, context.t0_frame, context.ego_id, context=context)
        if scene is None:
            raise RuntimeError('history-selected scene disappeared: ' + str(reason))
        eligible = bool(scene.future_observed_mask.all())
        counts['histories'] += 1
        counts['complete' if eligible else 'incomplete'] += 1
        if not eligible:
            continue
        label = scene_occupancy_pet(scene.future_native, scene.future_observed_mask, scene.dimensions,
                                    times=np.arange(175) * .04, ego_index=0, sample_period=.04, window=(0., 6.96),
                                    cap_seconds=4.)
        if not label['complete'] or not label['point_identified'] or not np.isfinite(label['pet_value_seconds']):
            raise RuntimeError('complete scene without a valid PET')
        complete.append(scene)
        labels.append(label)
        kept.append(item)
        indices.append(index)
        counts[label['status']] += 1
    metadata = dict(protocol='pcontrol_complete_scenes_v1', role=role, source_recording=recording.recording_id,
                    metric_protocol=PET_PROTOCOL, history_candidates=len(cohort))
    arrays = scene_arrays(complete, labels, kept, role=role, metadata=metadata)
    arrays['cohort_row_index'] = np.asarray(indices, dtype=np.int64)
    validate_complete_labels(arrays, complete_eligibility(arrays))
    return arrays, dict(counts), cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--raw-root', required=True, help='folder with the highD CSV files')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--recordings', nargs='+', help='recording ids, e.g. 35 47')
    group.add_argument('--split', choices=('train', 'val', 'test'), help='recordings of a split in configs/highd_split_v1.json')
    parser.add_argument('--out', required=True, help='output folder')
    parser.add_argument('--role', default='TEST', help='label stored with the scenes (at most five characters)')
    parser.add_argument('--quota', type=int, default=QUOTA, help='history candidates per recording (paper: 4096)')
    parser.add_argument('--salt', default=SALT, help='hash salt of the candidate order (paper: natural_ego_scene_v1)')
    args = parser.parse_args()
    if args.split:
        split = json.loads((ROOT / 'configs/highd_split_v1.json').read_text())
        recordings = split[args.split] if args.split in split else split['splits'][args.split]
    else:
        recordings = args.recordings
    out = Path(args.out)
    summary = {}
    for rec in recordings:
        recording = load_recording(args.raw_root, rec)
        selected, audit = select_history_cohort(recording, quota=args.quota, salt=args.salt)
        arrays, counts, cohort = complete_scenes(recording, selected, args.role)
        folder = out / recording.recording_id
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'complete_scenes.npz').open('wb') as handle:
            np.savez_compressed(handle, **arrays)
        # Every history candidate in selection order, before the future was inspected.
        (folder / 'history_cohort.json').write_text(json.dumps(json_safe(cohort)) + '\n')
        summary[recording.recording_id] = dict(counts, scenes=int(len(arrays['scene_id'])), selection=json_safe(audit))
        print(json.dumps(dict(recording=recording.recording_id, scenes=int(len(arrays['scene_id'])), **counts)), flush=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
