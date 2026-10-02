#!/usr/bin/env python3
"""Permission-fenced final natural validation; no command creates an approval."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.publication_pipeline.final_validation_protocol import (
    PROTOCOL,bind,verified_json,resolve,digest,same_binding,require_approval)

PREP=ROOT/'outputs/natural_percentile/transformer_publication_v1_20260916/final_validation_v1_preparation'
POLICY=ROOT/'configs/natural_percentile/final_natural_validation_v1.json'
OWN_CODE=(
    'pcontrol/research/run_final_natural_validation.py','pcontrol/research/rehearse_final_validation.py','pcontrol/research/check_final_generator_replay.py',
    'pcontrol/publication_pipeline/final_validation_protocol.py',
    'pcontrol/publication_pipeline/final_scene_data.py',
    'pcontrol/publication_pipeline/final_reference_suite.py',
    'pcontrol/publication_pipeline/final_generation_adapter.py',
    'pcontrol/publication_pipeline/final_execution_data.py',
    'pcontrol/publication_pipeline/final_execution_reference.py',
    'pcontrol/publication_pipeline/final_execution_generation.py',
    'pcontrol/publication_pipeline/final_execution_analysis.py',
    'pcontrol/research/check_final_reference_cuda_v2.py',
    'pcontrol/data/highd.py','pcontrol/data/clips.py','pcontrol/data/scenes.py',
    'pcontrol/data/scene_pet.py','pcontrol/data/complete_scene_view.py',
    'pcontrol/research/audit_guarded_generation.py','pcontrol/research/audit_transformer_reference_validation.py','pcontrol/research/audit_scene_context_diagnostics.py',
    'pcontrol/research/audit_time_attention_full_pipeline.py','pcontrol/research/analyze_guarded_physical_residuals.py',
    'pcontrol/publication_pipeline/generation_diagnostics.py')


def freeze_execution():
    from pcontrol.publication_pipeline.final_scene_data import write_once
    core_binding=bind(PREP/'frozen_contract.json');core=verified_json(core_binding)
    proof_binding=bind(PREP/'end_to_end_open_STOP04_v1/result.json');proof=verified_json(proof_binding)
    replay_binding=bind(PREP/'generator_adapter_replay.json');replay=verified_json(replay_binding)
    cuda_binding=bind(PREP/'reference_cuda_check_v2.json');cuda=verified_json(cuda_binding)
    if cuda['status']!='pass_declared_CUDA_reference_runtime' or cuda['maximum_CPU_CUDA_probability_difference']>cuda['acceptance_tolerance']:
        raise ValueError('declared reference CUDA runtime must pass unchanged numerical tolerance')
    if digest(cuda['code']['path'])!=cuda['code']['sha256']:raise ValueError('CUDA checker changed after its run')
    for value in replay['code'].values():
        if digest(value['path'])!=value['sha256']:raise ValueError('sampler replay source changed')
    extra_binding=bind(PREP/'end_to_end_open_STOP04_v1/analysis_supplement.json');extra=verified_json(extra_binding)
    if extra['status']!='complete' or extra['unique_natural_futures']['histories']!=3:
        raise ValueError('complete natural-context and failure-aware statistical analysis required')
    if (proof['status']!='pass_open_data_end_to_end_software_rehearsal' or proof['protected_data_read']
            or proof['actual_generated_requests']!=90 or proof['final_validation_performance']
            or not same_binding(proof['contract'],core_binding)):
        raise ValueError('complete open-data end-to-end QA required')
    if replay['status']!='pass_exact_P5_generator_replay' or any(r['maximum_trajectory_error']!=0 for r in replay['checks']):
        raise ValueError('exact prior-generator replay required')
    audit=verified_json(proof['generation_audit']);analysis=verified_json(proof['generation_analysis'])
    if audit['status']!='pass' or audit['retained_failures'] or audit['replayed_complete']!=90 or analysis['status']!='complete':
        raise ValueError('complete software trajectory audit required')
    policy=verified_json(core['policy'])
    codes=dict(core['code_sha256'])
    for source in (core['inherited_generator_code_sha256'],
                   verified_json(policy['sources']['guarded_reference_results'])['code_sha256'],
                   verified_json(policy['sources']['capacity_reference_results'])['code_sha256']):
        for path,value in source.items():
            if path in codes and codes[path]!=value:raise ValueError('inconsistent inherited code binding')
            codes[path]=value
    for path,value in codes.items():
        if digest(path)!=value:raise ValueError('inherited source drift: '+path)
    codes.update({path:digest(path) for path in OWN_CODE})
    import numpy,torch,scipy
    result=copy.deepcopy(core)
    result.update(status='execution_frozen_awaiting_explicit_authorization',model_contract=core_binding,
        execution_ready=True,execution_adapters_pending=[],execution_code_sha256=codes,
        software_rehearsal=proof_binding,generator_replay=replay_binding,
        CUDA_reference_check=cuda_binding,analysis_supplement=extra_binding,
        CUDA_GRU_inference_backend='native_PyTorch_no_cudnn_fusion',
        environment=dict(python=sys.version.split()[0],numpy=numpy.__version__,torch=torch.__version__,scipy=scipy.__version__),
        no_permission_generated=True,authorization_received=False,protected_raw_or_cached_trajectories_read=False)
    out=write_once(PREP/'execution_freeze_v1.json',result)
    print(json.dumps(dict(status=result['status'],execution_freeze=out,authorization_received=False)),flush=True)


def authorized_context(policy_binding,freeze_binding,approval_binding,scope):
    # The first operation in every production entry point is this gate.
    grant=require_approval(policy_binding,freeze_binding,approval_binding,scope=scope)
    frozen=verified_json(freeze_binding);policy=verified_json(policy_binding)
    if 'execution_code_sha256' not in frozen or not frozen.get('model_contract'):
        raise ValueError('complete execution freeze, not merely a model registry, required')
    for path,expected in frozen['execution_code_sha256'].items():
        if digest(path)!=expected:raise ValueError('execution code drift: '+path)
    import numpy,torch,scipy
    env=dict(python=sys.version.split()[0],numpy=numpy.__version__,torch=torch.__version__,scipy=scipy.__version__)
    if env!=frozen['environment']:raise ValueError('numerical runtime changed after execution freeze')
    root=resolve(policy['evaluation_root'])/scope
    if not root.is_relative_to(ROOT/'outputs/natural_percentile'):raise ValueError('outputs outside project evaluation namespace')
    return grant,policy,frozen,root


def run_stage(command,policy_binding,freeze_binding,approval_binding,scope,*,recording=None,arm=None):
    grant,p,f,root=authorized_context(policy_binding,freeze_binding,approval_binding,scope)
    from pcontrol.publication_pipeline.final_scene_data import (
        load_authorized_final_recording,select_history_cohort,extract_complete_recording,write_once)
    from pcontrol.publication_pipeline.final_execution_data import (
        finalize_dataset,CompleteFinalDataset,prepare_queue)
    root.mkdir(parents=True,exist_ok=True)
    if command=='build-recording':
        if recording not in grant.recording_ids:raise PermissionError('recording not covered by this approval')
        target=root/'data'/recording
        if target.exists():raise FileExistsError('do not repeat or overwrite an already started recording extraction')
        raw,sources,_=load_authorized_final_recording(policy_binding,freeze_binding,approval_binding,scope=scope,recording=recording)
        selected,selection=select_history_cohort(raw,quota=p['data']['history_candidates_per_recording'],salt=p['data']['history_hash_salt'])
        _,out=extract_complete_recording(raw,selected,output_dir=target,role=scope,selection_audit=selection,
            provenance=dict(contract=freeze_binding,authorization=approval_binding,raw_csvs=sources,
                development_rehearsal=False,protected_data_access=True,not_final_validation=False))
        return out
    if command=='finalize-data':
        bindings=[bind(root/'data'/rec/'recording_result.json') for rec in grant.recording_ids]
        return finalize_dataset(bindings,scope=scope,expected_recordings=list(grant.recording_ids),
            contract_binding=freeze_binding,output=root/'dataset.json')
    if command in ('reference','queue'):
        dataset=CompleteFinalDataset(bind(root/'dataset.json'),scope=scope,allowed_recordings=grant.recording_ids)
        if not same_binding(dataset.manifest['contract'],freeze_binding):raise ValueError('dataset belongs to another execution freeze')
        if command=='queue':return prepare_queue(dataset,freeze_binding,root/'queue')
        from pcontrol.publication_pipeline.final_generation_adapter import initialize_runtime
        initialize_runtime()
        import torch
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        from pcontrol.publication_pipeline.final_execution_reference import evaluate_references
        return evaluate_references(dataset,freeze_binding,root/'reference',device=p['runtime']['reference_device'])
    from pcontrol.publication_pipeline.final_generation_adapter import initialize_runtime
    from pcontrol.publication_pipeline.final_execution_generation import run_generation,audit_generation,describe_generation
    initialize_runtime();queue=bind(root/'queue'/'queue.json');q=verified_json(queue)
    if q['scope']!=scope or q['software_rehearsal'] or not same_binding(q['contract'],freeze_binding):raise ValueError('wrong final queue')
    if command=='generate':return run_generation(queue,freeze_binding,root/'generation',arm)
    if command=='audit':
        results={a:bind(root/'generation'/a/'result.json') for a in p['generation']['arms']}
        audit=audit_generation(queue,freeze_binding,results,root/'generation_audit.json')
        analysis=describe_generation(queue,results,audit,root/'generation_analysis.json')
        from pcontrol.publication_pipeline.final_execution_analysis import supplement_analysis
        extra=supplement_analysis(queue,results,analysis,root/'analysis_supplement.json')
        reference=bind(root/'reference'/'result.json');r=verified_json(reference)
        if not same_binding(r['contract'],freeze_binding):raise ValueError('reference result used another frozen contract')
        return write_once(root/'result.json',dict(status='complete_for_declared_cohort',scope=scope,contract=freeze_binding,
            authorization=approval_binding,dataset=bind(root/'dataset.json'),reference=reference,
            queue=queue,generation=results,generation_audit=audit,generation_analysis=analysis,
            analysis_supplement=extra,
            frozen_once_no_model_or_policy_selection=True,never_used_before_claim=False))
    raise ValueError('unknown final stage')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['freeze-execution','build-recording','finalize-data','reference','queue','generate','audit'])
    parser.add_argument('--policy',default=str(POLICY));parser.add_argument('--policy-sha256')
    parser.add_argument('--freeze',default=str(PREP/'execution_freeze_v1.json'));parser.add_argument('--freeze-sha256')
    parser.add_argument('--approval');parser.add_argument('--approval-sha256')
    parser.add_argument('--scope',choices=['VAL','TEST','R18']);parser.add_argument('--recording')
    parser.add_argument('--arm',choices=['canonical','atom_aware']);args=parser.parse_args()
    if args.command=='freeze-execution':return freeze_execution()
    pb=dict(path=args.policy,sha256=args.policy_sha256 or '')
    fb=dict(path=args.freeze,sha256=args.freeze_sha256 or '')
    ab=None if args.approval is None else dict(path=args.approval,sha256=args.approval_sha256 or '')
    try:out=run_stage(args.command,pb,fb,ab,args.scope,recording=args.recording,arm=args.arm)
    except PermissionError as exc:
        print(json.dumps(dict(status='blocked_before_protected_access',reason=str(exc),protected_data_read=False)),flush=True)
        raise SystemExit(2)
    print(json.dumps(dict(stage=args.command,scope=args.scope,result=out)),flush=True)


if __name__=='__main__':main()
