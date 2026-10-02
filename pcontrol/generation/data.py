"""Natural generation supervision with explicit FIT/STOP-only source access.

The archive members are decoded only for authorized role recordings. Every
example uses the same complete E_full scene roster, all actors and native
future samples. CAL/AUDIT constructors are deliberately rejected here.
"""
from functools import lru_cache
import hashlib
import json
import numpy as np

from pcontrol.data.complete_scene_view import verify_binding
from pcontrol.data.expanded_scene_view import ExpandedSceneReferenceView


DATA_SHA256 = "e05bd02e52a81ece015d86be8e106d7c353db42cc29377df364d9306d88c9012"
PREPARED_SHA256 = "d81c62b7459e71b6d4533b7c06e726afe6c797a6170cacfac2ae7fa78eccd613"
PROTOCOL = "natural_complete_scene_generation_supervision_v1"
DIAGNOSTIC_SALT = "natural_cosine_basis_FIT_reconstruction_v1"


class NaturalTrajectorySource:
    def __init__(self, manifest_binding, *, role):
        if role not in ("FIT", "STOP"):
            raise PermissionError("generation preparation cannot open CAL or AUDIT content")
        if manifest_binding["sha256"] != DATA_SHA256:
            raise ValueError("only the frozen natural complete-scene source is allowed")
        self.view = ExpandedSceneReferenceView(manifest_binding["path"], manifest_binding["sha256"],
            purpose="fit" if role == "FIT" else "stop")
        self.role, self.rows, self.binding = role, self.view.rows, manifest_binding
        if len(self.rows) != (9913 if role == "FIT" else 538):
            raise ValueError("the immutable FIT/STOP complete-scene denominator changed")

    def __len__(self): return len(self.rows)

    @lru_cache(maxsize=2)
    def _future_record(self, recording):
        entry = self.view.manifest["recordings"][recording]
        if entry["role"] != self.role:
            raise PermissionError("recording role changed before future archive decode")
        path = verify_binding(entry["data"], self.view.paths[recording])
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("scene_id", "recording_id", "role", "offsets", "agent_ids",
                "future_native_agents", "future_observed_mask_agents", "history_frame_ids", "future_frame_ids")}
        verify_binding(entry["data"], path)
        if not np.all(arrays["role"] == self.role) or not np.all(arrays["recording_id"] == recording):
            raise ValueError("actual future archive belongs to a different role")
        return arrays

    def __getitem__(self, index):
        rec, row, sid = self.rows[index]
        example = self.view[index]
        arrays = self._future_record(rec)
        lo, hi = map(int, arrays["offsets"][row:row+2])
        if str(arrays["scene_id"][row]) != sid:
            raise ValueError("future supervision identity differs from the frozen history view")
        future = arrays["future_native_agents"][lo:hi].transpose(1,0,2).copy()
        mask = arrays["future_observed_mask_agents"][lo:hi].T
        history = example["features"]["history"]
        if (future.shape != (175, history.shape[1], 4) or not mask.all()
                or not np.isfinite(future).all() or not np.array_equal(future[0], history[-1])
                or not np.all(np.diff(arrays["future_frame_ids"][row]) == 1)
                or not np.all(np.diff(arrays["history_frame_ids"][row]) == 2)
                or arrays["future_frame_ids"][row,0] != arrays["history_frame_ids"][row,-1]):
            raise ValueError("actual H/t0/full native future consistency failed")
        return dict(features=example["features"], future_observed=future, anchors=history[-1].copy(),
            agent_ids=arrays["agent_ids"][lo:hi].copy(), target_ref=float(example["target"]),
            metadata=example["metadata"], source_binding=self.view.manifest["recordings"][rec]["data"],
            future_frame_ids=arrays["future_frame_ids"][row].copy())


def diagnostic_indices(source, maximum=128):
    if source.role != "FIT" or type(maximum) is not int or not 1 <= maximum <= 128:
        raise ValueError("basis diagnosis uses at most128 FIT scenes, never STOP/CAL/AUDIT")
    return sorted(range(len(source)), key=lambda i: (
        hashlib.sha256((DIAGNOSTIC_SALT+"|"+source.rows[i][2]).encode()).hexdigest(), source.rows[i][2]))[:maximum]


FEATURE_KEYS = frozenset(("history", "dimensions", "road_boundaries", "road_boundary_mask", "ego_mask", "agent_mask"))
PACK_KEYS = FEATURE_KEYS | {"coef_clean", "anchors", "scene_id", "recording_id", "role"}


def fit_coefficient_normalizer(coefficients, scene_ids, *, role):
    if role != "FIT" or not coefficients or len(coefficients) != len(scene_ids):
        raise ValueError("coefficient mean/std require only nonempty FIT scene coefficients")
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("FIT normalization scene identities must be unique")
    modes = coefficients[0].shape[-2]
    if any(c.ndim != 3 or c.shape[0] < 3 or c.shape[1:] != (modes,2) or not np.isfinite(c).all() for c in coefficients):
        raise ValueError("normalize only actual unpadded agents and finite physical coefficients")
    values = np.concatenate(coefficients, axis=0).astype(np.float64)
    scale = np.maximum(values.std(axis=0), 1e-6)
    return dict(protocol="natural_FIT_coefficient_mean_std_v1", mean=values.mean(axis=0).tolist(),
        scale=scale.tolist(), scale_is_floored_population_std=True, std_floor=1e-6, centered=True,
        fit_scene_count=len(scene_ids), fit_valid_agent_count=len(values), roles_used=["FIT"],
        FIT_scene_id_sha256=hashlib.sha256("\n".join(scene_ids).encode()).hexdigest(),
        STOP_CAL_AUDIT_used=False, labels_or_risk_used=False,
        coefficient_layout="agent,mode,xy", padding_used_for_statistics=False)


def pack_training_examples(examples, coefficients, coefficient_normalizer, history_normalizer):
    if not examples or len(examples) != len(coefficients):
        raise ValueError("matching natural examples and encoded coefficients required")
    roles = {e["metadata"]["role"] for e in examples}
    if len(roles) != 1 or not roles <= {"FIT", "STOP"}:
        raise PermissionError("a training pack is exactly FIT or STOP, never mixed/heldout")
    mean, scale = np.asarray(coefficient_normalizer["mean"]), np.asarray(coefficient_normalizer["scale"])
    if mean.shape != scale.shape or mean.ndim != 2 or mean.shape[1] != 2 or np.any(scale <= 0):
        raise ValueError("mode-by-xy mean/std required")
    modes, batch = mean.shape[0], len(examples)
    counts = [e["features"]["history"].shape[1] for e in examples]
    road_counts = [len(e["features"]["road_boundaries"]) for e in examples]
    maximum, roads = max(counts), max(road_counts)
    hscale, dscale = np.asarray(history_normalizer["history_scale"]), np.asarray(history_normalizer["dimension_scale"])
    if (history_normalizer["roles_used"] != ["FIT"] or history_normalizer["history_centering"] is not False
            or history_normalizer["future_or_CAL_or_AUDIT_used"] is not False):
        raise ValueError("only frozen FIT history scale-only normalization may be reused")
    result = dict(history=np.zeros((batch,13,maximum,4),np.float32),
        dimensions=np.zeros((batch,maximum,2),np.float32), road_boundaries=np.zeros((batch,roads),np.float32),
        road_boundary_mask=np.zeros((batch,roads),bool),ego_mask=np.zeros((batch,maximum),bool),
        agent_mask=np.zeros((batch,maximum),bool),coef_clean=np.zeros((batch,maximum,modes,2),np.float32),
        anchors=np.zeros((batch,maximum,4),np.float64))
    for i,(example,c) in enumerate(zip(examples,coefficients)):
        f=example["features"];n,r=counts[i],road_counts[i]
        if c.shape != (n,modes,2) or example["anchors"].shape != (n,4):
            raise ValueError("coefficients/anchors must preserve all actual history-selected agents")
        result["history"][i,:,:n]=f["history"]/hscale
        result["dimensions"][i,:n]=f["dimensions"]/dscale
        result["road_boundaries"][i,:r]=f["road_boundaries"]/hscale[1]
        result["road_boundary_mask"][i,:r]=True
        result["ego_mask"][i,:n]=f["ego_mask"]
        result["agent_mask"][i,:n]=True
        result["coef_clean"][i,:n]=(c-mean)/scale
        result["anchors"][i,:n]=example["anchors"]
    result.update(scene_id=np.asarray([e["metadata"]["scene_id"] for e in examples]),
        recording_id=np.asarray([e["metadata"]["recording_id"] for e in examples]),
        role=np.asarray([e["metadata"]["role"] for e in examples]))
    if set(result) != PACK_KEYS:
        raise RuntimeError("generation pack exposed undeclared fields")
    return result


def load_training_pack(binding, *, role):
    if role not in ("FIT","STOP"):
        raise PermissionError("training packs cannot decode CAL/AUDIT")
    path=verify_binding(binding)
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files) != PACK_KEYS:
            raise ValueError("generation pack contains unexpected future/label/input fields")
        declared=archive["role"]
        if not np.all(declared==role):raise PermissionError("role checked before decoding training tensors")
        arrays={key:archive[key] for key in PACK_KEYS if key!="role"};arrays["role"]=declared
    verify_binding(binding,path)
    if (arrays["agent_mask"].dtype != np.bool_ or not np.all(arrays["agent_mask"].sum(1)>=3)
            or arrays["coef_clean"].shape[:2] != arrays["agent_mask"].shape
            or arrays["anchors"].shape != arrays["agent_mask"].shape+(4,)
            or not np.isfinite(arrays["coef_clean"]).all() or not np.isfinite(arrays["anchors"]).all()):
        raise ValueError("invalid all-agent coefficients/masks/anchors")
    return arrays
