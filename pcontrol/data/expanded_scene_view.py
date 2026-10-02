"""FIT-expanded E_full view with unchanged STOP/CAL/AUDIT observations.

Only actual history, sizes, road geometry and ego role enter features. Native
future masks certify dataset eligibility, then are removed from loaded arrays;
native future trajectories and raw CSVs are never decoded by this reader.
"""
from functools import lru_cache
import json

import numpy as np

from .complete_scene_view import (
    CompleteSceneReferenceView as OriginalCompleteSceneReferenceView,
    ELIGIBILITY, ESTIMAND, ROLE_PURPOSE, SOURCE_SHA256,
    collate_reference_examples, complete_eligibility, reference_example,
    resolve, validate_complete_labels, verify_binding,
)


PROTOCOL = "natural_expanded_complete_scene_reference_view_v1"
RECORD_PROTOCOL = "natural_expanded_complete_scene_recording_v1"
BASE_VIEW_SHA256 = "85d48d841047969264816cbe84e46e11ccabc46f0bdf003ce5c9b0d5cba23d9a"
CONFIG_SHA256 = "a920b35d8a06da8efda7d78e5c4fc203ee88cbdfde1bf265ae55d1e91a8e79f8"
FEATURE_KEYS = frozenset(("history", "dimensions", "road_boundaries", "ego_mask"))
READ_KEYS = ("offsets", "future_observed_mask_agents", "complete_horizon", "point_identified",
             "pet_value", "label_status", "label_interval_seconds", "scene_id", "recording_id",
             "role", "history_agents", "dimensions_agents", "carriageway_boundaries",
             "carriageway_boundary_mask", "cohort_row_index", "metadata_json")


def _bound_json(binding):
    return json.loads(verify_binding(binding).read_text())


def validate_manifest(manifest):
    """Authenticate metadata without opening any raw observations or future arrays."""
    if (manifest.get("protocol") != PROTOCOL or manifest.get("status") != "complete"
            or manifest.get("eligibility") != ELIGIBILITY or manifest.get("estimand") != ESTIMAND
            or manifest.get("base_complete_view", {}).get("sha256") != BASE_VIEW_SHA256
            or manifest.get("source_scene_manifest", {}).get("sha256") != SOURCE_SHA256
            or manifest.get("expansion_config_sha256") != CONFIG_SHA256
            or manifest.get("FIT_history_quota") != 4096 or manifest.get("loss") != "CRPS"):
        raise ValueError("not the frozen FIT-only complete-scene expansion")
    for key in ("partial_zero_included", "future_availability_is_model_input", "simulation_or_generated_futures",
                "generator_training", "target_cohort_is_all_original_scenes", "missing_data_population_recovery",
                "model_training_executed_by_expansion"):
        if manifest.get(key) is not False:
            raise ValueError("invalid expansion scope: " + key)
    for key in ("nonFIT_views_bitwise_unchanged", "nested_original512_verified", "no_future_critical_actor_input"):
        if manifest.get(key) is not True:
            raise ValueError("missing expansion guarantee: " + key)
    if not manifest.get("code_sha256") or not manifest.get("expansion_worker_code_sha256"):
        raise ValueError("source-code hashes required")
    for path, checksum in manifest["code_sha256"].items():
        verify_binding(dict(path=path, sha256=checksum))
    expected_code = dict(manifest["expansion_worker_code_sha256"])
    loader_key = "pcontrol/data/expanded_scene_view.py"
    if loader_key in expected_code or loader_key not in manifest["code_sha256"]:
        raise ValueError("worker and completed-view code boundaries differ")
    expected_code[loader_key] = manifest["code_sha256"][loader_key]
    if manifest["code_sha256"] != expected_code:
        raise ValueError("completed-view dependency set differs from frozen workers")
    base = _bound_json(manifest["base_complete_view"])
    source = _bound_json(manifest["source_scene_manifest"])
    roles = _bound_json(manifest["input_bindings"]["roles"])
    original = _bound_json(manifest["input_bindings"]["original_split"])
    if (manifest["input_bindings"]["complete_view"] != manifest["base_complete_view"]
            or manifest["input_bindings"]["scene_manifest"] != manifest["source_scene_manifest"]
            or set(manifest["recordings"]) != set(base["recordings"])
            or set(manifest["recordings"]) != set(original["splits"]["train"])):
        raise ValueError("authorized original29 roster or parent bindings changed")
    ids = []
    totals = {role: 0 for role in ROLE_PURPOSE.values()}
    fit_records = set()
    for rec, entry in manifest["recordings"].items():
        old = base["recordings"][rec]
        role = entry["role"]
        if role != old["role"] or rec not in roles[role]:
            raise ValueError("a recording changed its immutable algorithm role")
        if len(entry["selected_rows"]) != len(entry["selected_scene_ids"]):
            raise ValueError("scene identities and rows differ in length")
        if role != "FIT":
            compared = dict(entry)
            if compared.pop("source_kind", None) != "unchanged_original_complete_view" or compared != old:
                raise ValueError("nonFIT rows, targets, ordering or bindings changed")
        else:
            fit_records.add(rec)
            if (entry.get("source_kind") != "expanded_FIT_complete_only"
                    or entry["selected_rows"] != list(range(len(entry["selected_scene_ids"])))
                    or entry["counts"]["complete_scenes"] != len(entry["selected_rows"])
                    or not 512 <= entry["counts"]["source_scenes"] <= 4096):
                raise ValueError("expanded FIT shard is not a complete-only nested cohort")
        totals[role] += len(entry["selected_rows"])
        ids.extend(entry["selected_scene_ids"])
    if fit_records != set(roles["FIT"]) or len(fit_records) != 13 or set(manifest["FIT_worker_results"]) != fit_records:
        raise ValueError("only the original FIT13 may have new shards")
    if len(set(ids)) != len(ids) or manifest["complete_scenes"] != len(ids):
        raise ValueError("duplicate scene identity or total disagreement")
    if set(manifest["by_role"]) != set(totals):
        raise ValueError("all four algorithm roles required")
    for role, total in totals.items():
        if manifest["by_role"][role]["complete_scenes"] != total:
            raise ValueError("role denominator disagrees with rows")
    if any(totals[role] != number for role, number in (("STOP", 538), ("CAL", 504), ("AUDIT", 487))):
        raise ValueError("nonFIT denominator changed")
    return base, source


class ExpandedSceneReferenceView:
    """Same public example API as the frozen complete view, one role per instance."""

    def __init__(self, manifest_path, manifest_sha256, *, purpose):
        if purpose not in ROLE_PURPOSE:
            raise ValueError("explicit fit/stop/calibrate/audit purpose required")
        path = verify_binding(dict(path=str(manifest_path), sha256=manifest_sha256))
        self.manifest = json.loads(path.read_text())
        self.base, self.source = validate_manifest(self.manifest)
        self.role = ROLE_PURPOSE[purpose]
        self._delegate = None
        if self.role != "FIT":
            binding = self.manifest["base_complete_view"]
            self._delegate = OriginalCompleteSceneReferenceView(binding["path"], binding["sha256"], purpose=purpose)
            self.rows, self.paths = self._delegate.rows, self._delegate.paths
            return
        self.rows, self.paths, self.results = [], {}, {}
        for rec, entry in self.manifest["recordings"].items():
            if entry["role"] != "FIT":
                continue
            if entry["result_binding"] != self.manifest["FIT_worker_results"][rec]:
                raise ValueError("worker result binding mismatch")
            result = _bound_json(entry["result_binding"])
            if (result.get("protocol") != RECORD_PROTOCOL or result.get("status") != "complete"
                    or result.get("recording_id") != rec or result.get("role") != "FIT"
                    or result.get("config_sha256") != CONFIG_SHA256
                    or result.get("input_bindings") != self.manifest["input_bindings"]
                    or result.get("code_sha256") != self.manifest["expansion_worker_code_sha256"]
                    or result.get("original512_nested_ids_history_rosters_eligibility_and_PET_exact") is not True
                    or result.get("incomplete_rows_replaced") is not False
                    or result.get("incomplete_PET_computed") is not False
                    or result.get("complete_only") is not True
                    or result.get("complete_scene_ids") != entry["selected_scene_ids"]):
                raise ValueError("expanded FIT worker lineage changed")
            for name in ("data", "cohort", "eligibility_ledger"):
                if entry[name] != result[name]:
                    raise ValueError("FIT artifact binding mismatch")
                verify_binding(entry[name])
            self.paths[rec] = verify_binding(entry["data"], path.parent / "FIT" / (rec + ".npz"))
            self.results[rec] = result
            self.rows.extend((rec, row, sid) for row, sid in zip(entry["selected_rows"], entry["selected_scene_ids"]))

    def __len__(self):
        return len(self.rows)

    @lru_cache(maxsize=2)
    def _record(self, rec):
        entry = self.manifest["recordings"][rec]
        verify_binding(entry["data"], self.paths[rec])
        with np.load(self.paths[rec], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in READ_KEYS}
        verify_binding(entry["data"], self.paths[rec])
        metadata = json.loads(str(arrays.pop("metadata_json")))
        if (metadata.get("protocol") != RECORD_PROTOCOL or metadata.get("config_sha256") != CONFIG_SHA256
                or metadata.get("input_bindings") != self.manifest["input_bindings"]
                or metadata.get("code_sha256") != self.manifest["expansion_worker_code_sha256"]
                or metadata.get("eligibility") != ELIGIBILITY or metadata.get("estimand") != ESTIMAND
                or metadata.get("complete_only") is not True or metadata.get("natural_only") is not True
                or metadata.get("focal_vehicle_input") is not False or metadata.get("partial_zero_included") is not False
                or metadata.get("simulation_or_generated_futures") is not False):
            raise ValueError("expanded FIT shard metadata changed")
        eligible = complete_eligibility(arrays)
        validate_complete_labels(arrays, eligible)
        if (not eligible.all() or arrays["scene_id"].tolist() != entry["selected_scene_ids"]
                or not np.all(arrays["recording_id"] == rec) or not np.all(arrays["role"] == "FIT")
                or arrays["cohort_row_index"].tolist() != self.results[rec]["complete_cohort_rows"]):
            raise ValueError("FIT membership differs from the selected complete cohort")
        del arrays["future_observed_mask_agents"]
        return arrays

    def __getitem__(self, index):
        if self._delegate is not None:
            return self._delegate[index]
        rec, row, sid = self.rows[index]
        result = reference_example(self._record(rec), row)
        if result["metadata"]["scene_id"] != sid or set(result["features"]) != FEATURE_KEYS:
            raise ValueError("reference feature boundary or identity differs")
        return result


CompleteSceneReferenceView = ExpandedSceneReferenceView
