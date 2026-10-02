# Data

## highD recordings
The experiments use the 60 recordings of the [highD dataset](https://levelxdata.com/highd-dataset/). `configs/highd_split_v1.json` assigns them to three groups by recording:

| Group | Recordings | Use in the paper |
| --- | --- | --- |
| `train` | 29 recordings: 01–06, 08, 10–12, 17, 20, 22, 25, 26, 28, 29, 36, 40, 41, 43, 46, 49, 51–53, 55, 57, 59 | fitting, early stopping, calibration and development of the reference and the generator |
| `val` | 14 recordings: 07, 09, 13–15, 30, 31, 37, 45, 48, 50, 56, 58, 60 | the additional evaluation set |
| `test` | 16 recordings: 16, 19, 21, 23, 24, 27, 32–35, 38, 39, 42, 44, 47, 54 | the primary evaluation set |

Recording 18 lies outside the split and forms the single-recording evaluation set. The roles of the training recordings (fitting, early stopping, calibration, development) are listed in `configs/natural_percentile/ego_scene_roles_v1.json`.

## Scenes
`pcontrol/tools/prepare_highd.py` builds scenes from one recording at a time.

1. **History candidates.** Every vehicle at every frame divisible by 25 (one candidate per second) is a candidate ego. The candidates are examined in the order of a salted hash of (recording, ego, frame) until 4,096 histories are found.
2. **History.** A candidate gives a history when the ego is a car or truck. The history holds the ego and every car or truck in the same driving direction within 120 m longitudinally that is observed over all 13 history states (0.96 s, every 0.08 s). It needs at least three vehicles.
3. **Future.** The future covers 175 states (6.96 s at 25 Hz, the first state being the last history state). A scene is kept when every vehicle of its history is observed over the whole future. Missing futures are never filled or replaced.
4. **Risk.** The surrogate is the minimum post-encroachment time (PET) between the ego and its surrounding vehicles over the future, computed from vehicle footprints and capped at 4 s. Its distribution has point masses at 0 and at the 4 s cap.

Coordinates are in meters and meters per second, in a frame whose origin is the ego position at the last history state and in which traffic moves toward +x.

## Arrays of `complete_scenes.npz`
Vehicle arrays are stacked over scenes. The vehicles of scene `i` are the rows `offsets[i]:offsets[i+1]`, the ego first.

| Array | Shape | Content |
| --- | --- | --- |
| `scene_id`, `recording_id`, `ego_id`, `t0_frame` | (S,) | identity of each scene |
| `num_agents`, `offsets` | (S,), (S+1,) | vehicles per scene and row offsets |
| `agent_ids` | (A,) | highD vehicle ids |
| `dimensions_agents` | (A, 2) | length and width |
| `history_agents` | (A, 13, 4) | x, y, vx, vy over the history |
| `future_native_agents` | (A, 175, 4) | x, y, vx, vy over the recorded future |
| `carriageway_boundaries`, `carriageway_boundary_mask` | (S, 4) | lateral positions of the lane markings of the ego's carriageway |
| `pet_value` | (S,) | minimum ego–SV PET of the recorded future (s) |
| `label_status` | (S,) | finite, capped, or no shared occupancy within the window |

`pcontrol.api.load_scenes` turns each scene into the model features (`history` of shape (13, N, 4), `dimensions`, `road_boundaries`, `ego_mask`, `agent_mask`), the recorded future and its PET.

On the `test` split, the tool gives the 11,649 scenes of the paper's primary evaluation set, identical to the arrays used in the paper.
