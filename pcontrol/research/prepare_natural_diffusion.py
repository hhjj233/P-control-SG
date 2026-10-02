#!/usr/bin/env python3
"""Prepare actual FIT/STOP-only, no-risk-label natural diffusion supervision."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[name]="1"
import numpy as np

from pcontrol.data.complete_scene_view import verify_binding,resolve,sha256
from pcontrol.data.scene_pet import scene_occupancy_pet,PROTOCOL as PET_PROTOCOL
from pcontrol.generation.trajectory_basis import TrajectoryBasis,basis_matrices,reconstruction_errors
from pcontrol.generation.data import (NaturalTrajectorySource,diagnostic_indices,DATA_SHA256,PREPARED_SHA256,
    DIAGNOSTIC_SALT,fit_coefficient_normalizer,pack_training_examples,load_training_pack)

PROTOCOL="natural_diffusion_training_data_v1"
CODE=("pcontrol/research/prepare_natural_diffusion.py","pcontrol/generation/trajectory_basis.py",
      "pcontrol/generation/data.py","pcontrol/data/expanded_scene_view.py",
      "pcontrol/data/complete_scene_view.py","pcontrol/data/scene_pet.py")


def write_json(path,value):
    with Path(path).open("x",encoding="utf-8") as handle:json.dump(value,handle,sort_keys=True,indent=2,allow_nan=False)
    return dict(path=str(Path(path).resolve()),sha256=sha256(path))


def save_npz(path,arrays):
    with Path(path).open("xb") as handle:np.savez_compressed(handle,**arrays)
    return dict(path=str(Path(path).resolve()),sha256=sha256(path))


def bound_json(binding):return json.loads(verify_binding(binding).read_text())


def aggregate_errors(rows):
    xy_count=sum(r["position_scalar_count"] for r in rows)
    xy=float(np.sqrt(sum(r["position_squared_error_sum"] for r in rows)/xy_count))
    return dict(scenes=len(rows),pooled_xy_coordinate_RMSE_m=xy,pooled_position_vector_RMSE_m=float(xy*np.sqrt(2)),
        scene_xy_coordinate_RMSE_p95_m=float(np.quantile([r["position_rmse_m"] for r in rows],.95)),
        scene_position_vector_RMSE_p95_m=float(np.quantile([r["position_vector_rmse_m"] for r in rows],.95)),
        pooled_velocity_coordinate_RMSE_mps=float(np.sqrt(sum(r["velocity_squared_error_sum"] for r in rows)/sum(r["velocity_scalar_count"] for r in rows))),
        scene_position_vector_RMSE_max_m=float(max(r["position_vector_rmse_m"] for r in rows)),
        all_t0_states_exact=all(r["t0_state_exact"] for r in rows))


def diagnose_basis(source):
    selected=diagnostic_indices(source,128)
    rows={k:[] for k in (8,16,24)}
    # Preserve hash order in the declaration; group source reads by recording.
    for number,index in enumerate(sorted(selected,key=lambda i:source.rows[i][:2])):
        example=source[index];observed=example["future_observed"];mask=np.ones(observed.shape[:2],bool)
        for k in rows:
            basis=TrajectoryBasis(k);coefs=basis.encode(observed,example["anchors"])
            decoded=basis.decode(coefs,example["anchors"]);error=reconstruction_errors(observed,decoded)
            label=scene_occupancy_pet(decoded,mask,example["features"]["dimensions"],times=basis.times,
                sample_period=.04,window=(0.,6.96),cap_seconds=4.,ego_index=0)
            if not label["complete"] or not label["point_identified"]:raise ValueError("a complete decoded trajectory has no exact PET")
            error.update(scene_id=example["metadata"]["scene_id"],recording_id=example["metadata"]["recording_id"],
                actual_agents=observed.shape[1],observed_PET=example["target_ref"],reconstructed_PET=label["pet_value_seconds"],
                PET_absolute_error_s=abs(label["pet_value_seconds"]-example["target_ref"]),
                source_future_sha256=hashlib.sha256(observed.tobytes()).hexdigest())
            rows[k].append(error)
        if (number+1)%32==0:print(json.dumps(dict(stage="FIT_basis_diagnosis",scenes=number+1,maximum=128)),flush=True)
    summaries={}
    for k,values in rows.items():
        summary=aggregate_errors(values)
        summary["PET_absolute_error_p95_s"]=float(np.quantile([r["PET_absolute_error_s"] for r in values],.95))
        summary["PET_absolute_error_max_s"]=float(max(r["PET_absolute_error_s"] for r in values))
        # Euclidean position RMSE is the conservative gate; coordinate RMSE
        # is also retained so the two conventions cannot be confused.
        summary["gate_passed"]=bool(summary["pooled_position_vector_RMSE_m"]<.10
            and summary["scene_position_vector_RMSE_p95_m"]<.15 and summary["PET_absolute_error_p95_s"]<.05
            and summary["all_t0_states_exact"])
        summaries[str(k)]=summary
    passing=[k for k in (8,16,24) if summaries[str(k)]["gate_passed"]]
    return dict(protocol="natural_FIT_only_basis_reconstruction_diagnostic_v1",dataset_manifest=source.binding,
        role="FIT",hash_salt=DIAGNOSTIC_SALT,selected_scene_ids=[source.rows[i][2] for i in selected],
        selected_K=min(passing) if passing else None,K_candidates=[8,16,24],per_K=summaries,per_scene=rows,
        gates=dict(pooled_position_vector_RMSE_lt_m=.10,scene_position_vector_RMSE_p95_lt_m=.15,
                   PET_absolute_error_p95_lt_seconds=.05,velocity_is_report_only=True),
        PET_metric_version=PET_PROTOCOL,uncapped_PET_not_substituted_for_capped_label=True,
        future_values_modified=False,reconstructed_futures_are_observations=False,
        CAL_AUDIT_STOP_used_for_basis_selection=False,maximum_unique_FIT_scenes_with_PET_evaluation=128)


def prepare(policy_binding,output_root,*,diagnose_only=False):
    policy=bound_json(policy_binding)
    if (policy["data_manifest"]["sha256"]!=DATA_SHA256 or policy["reference_prepared"]["sha256"]!=PREPARED_SHA256
            or policy["generator"]["history_conditioned_only"] is not True
            or policy["generator"]["risk_labels_used_in_training"] is not False
            or policy["generator"]["focal_or_future_semantics_input"] is not False):
        raise ValueError("only the fixed history-only natural-data generator protocol is accepted")
    ref=bound_json(policy["reference_prepared"]);hnorm=bound_json(ref["normalizer"])
    if ref["data"]["sha256"]!=DATA_SHA256 or hnorm["fit_scene_count"]!=9913 or hnorm["roles_used"]!=["FIT"]:
        raise ValueError("H normalizer does not belong to this exact natural FIT9913 population")
    source=NaturalTrajectorySource(policy["data_manifest"],role="FIT")
    fit_ids=[row[2] for row in source.rows]
    if hashlib.sha256("\n".join(fit_ids).encode()).hexdigest()!=hnorm["FIT_scene_id_sha256"]:
        raise ValueError("reused history normalizer was fitted on different scene identities")
    output=resolve(output_root);output.mkdir(parents=True,exist_ok=False)
    code={path:sha256(ROOT/path) for path in CODE}
    write_json(output/"source_access.json",dict(protocol=PROTOCOL,policy=policy_binding,dataset_manifest=source.binding,
        allowed_content_roles=["FIT","STOP"],CAL_AUDIT_content_opened=False,raw_CSV_read=False,
        whole_selected_role_archive_members_may_include_unselected_rows=True,
        actual_supervision_uses_only_immutable_E_full_rows=True,code_sha256=code))
    diagnostic=diagnose_basis(source)
    diagnostic["code_sha256"]=code
    diagnostic_binding=write_json(output/"basis_diagnostic.json",diagnostic)
    if diagnostic["selected_K"] is None:raise RuntimeError("all predefined representation gates failed; no pack or training produced")
    if diagnose_only:return dict(status="diagnosed",diagnostic=diagnostic_binding,selected_K=diagnostic["selected_K"])
    basis=TrajectoryBasis(diagnostic["selected_K"])
    collections={};reconstruction={};role_sources={"FIT":source};provenance={};reference_statistics={}
    normalizer=None;normalizer_binding=None;packs={};counts={}
    for role in ("FIT","STOP"):
        if role=="STOP":role_sources[role]=NaturalTrajectorySource(policy["data_manifest"],role="STOP")
        current=role_sources[role];examples=[];coefs=[];errors=[];targets=[]
        ledger=output/(role+"_source_rows.jsonl")
        with ledger.open("x",encoding="utf-8") as handle:
            for i in range(len(current)):
                item=current[i];c=basis.encode(item["future_observed"],item["anchors"])
                err=reconstruction_errors(item["future_observed"],basis.decode(c,item["anchors"]))
                errors.append(err);coefs.append(c);targets.append(item["target_ref"])
                examples.append(dict(features=item["features"],anchors=item["anchors"],metadata=item["metadata"]))
                record=dict(item["metadata"],actual_agent_ids=item["agent_ids"].tolist(),num_agents=len(item["agent_ids"]),
                    source_binding=item["source_binding"],native_future_frame_ids=item["future_frame_ids"].tolist(),
                    actual_future_sha256=hashlib.sha256(item["future_observed"].tobytes()).hexdigest())
                handle.write(json.dumps(record,sort_keys=True,allow_nan=False)+"\n")
                if (i+1)%1024==0:print(json.dumps(dict(stage="encode_actual_natural_future",role=role,scenes=i+1,total=len(current))),flush=True)
        reconstruction[role]=aggregate_errors(errors)
        if role=="FIT":
            if (reconstruction[role]["pooled_position_vector_RMSE_m"]>=.10
                    or reconstruction[role]["scene_position_vector_RMSE_p95_m"]>=.15):
                raise RuntimeError("full-FIT position reconstruction gate failed; no next-K/threshold change is automatic")
            normalizer=fit_coefficient_normalizer(coefs,fit_ids,role="FIT")
            normalizer_binding=write_json(output/"coefficient_normalizer.json",normalizer)
        pack=pack_training_examples(examples,coefs,normalizer,hnorm)
        packs[role]=save_npz(output/(role+"_generator_pack.npz"),pack)
        loaded=load_training_pack(packs[role],role=role)
        if len(loaded["scene_id"])!=len(current):raise ValueError("saved training pack lost a scene")
        counts[role]=len(current)
        provenance[role]=dict(path=str(ledger),sha256=sha256(ledger))
        reference_statistics[role]=dict(label_is_model_input=False,used_to_fit_coefficients=False,
            minimum=float(min(targets)),maximum=float(max(targets)),mean=float(np.mean(targets)),
            zero_count=int(np.sum(np.asarray(targets)==0)),cap_count=int(np.sum(np.asarray(targets)==4)))
        del pack,loaded,examples,coefs
    bp,bv,ba=basis_matrices(basis.times,basis.modes,basis.horizon)
    matrices=save_npz(output/"basis_matrices.npz",dict(times=basis.times,position=bp,velocity=bv,acceleration=ba))
    for path,digest in code.items():verify_binding(dict(path=path,sha256=digest))
    manifest=dict(protocol=PROTOCOL,status="prepared",policy=policy_binding,dataset_manifest=source.binding,
        reference_prepared=policy["reference_prepared"],packs=packs,basis=basis.as_dict(),basis_matrices=matrices,
        coefficient_normalizer=normalizer_binding,history_normalizer=ref["normalizer"],basis_diagnostic=diagnostic_binding,
        counts=counts,source_rows=provenance,full_role_position_reconstruction=reconstruction,
        original_PET_reference_statistics_only=reference_statistics,code_sha256=code,
        FIT_recordings=sorted({row[0] for row in role_sources["FIT"].rows}),
        STOP_recordings=sorted({row[0] for row in role_sources["STOP"].rows}),
        coefficient_layout="B,N,K,2",anchor_layout="B,N,4 actual xy0/v0 float64",feature_dtype="float32",
        no_native_future_or_PET_in_training_packs=True,actual_future_supervision_retained_by_immutable_source_binding=True,
        risk_labels_used_for_training=False,all_actual_agents_retained=True,padding_is_not_real_traffic=True,
        CAL_decoded=False,AUDIT_decoded=False,raw_CSV_read=False,generator_training_executed=False,
        simulated_futures_used=False,reconstructed_futures_claimed_as_observations=False)
    binding=write_json(output/"manifest.json",manifest)
    print(json.dumps(dict(stage="natural_generation_data_prepared",manifest=binding,counts=counts,K=basis.modes)),flush=True)
    return binding


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--policy",required=True);parser.add_argument("--policy-sha256",required=True)
    parser.add_argument("--output-root",required=True);parser.add_argument("--diagnose-only",action="store_true")
    args=parser.parse_args();binding=dict(path=str(resolve(args.policy)),sha256=args.policy_sha256)
    prepare(binding,args.output_root,diagnose_only=args.diagnose_only)


if __name__=="__main__":main()
