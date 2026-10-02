#!/usr/bin/env python3
"""End-to-end final-adapter QA on three previously open natural STOP04 clips."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.publication_pipeline.final_validation_protocol import bind,verified_json
from pcontrol.publication_pipeline.final_execution_data import finalize_dataset,CompleteFinalDataset,prepare_queue
from pcontrol.publication_pipeline.final_execution_reference import evaluate_references
from pcontrol.publication_pipeline.final_execution_generation import run_generation,audit_generation,describe_generation
from pcontrol.publication_pipeline.final_generation_adapter import initialize_runtime
from pcontrol.publication_pipeline.final_scene_data import write_once


def run():
    initialize_runtime();start=time.monotonic()
    prep=ROOT/'outputs/natural_percentile/transformer_publication_v1_20260916/final_validation_v1_preparation'
    contract=bind(prep/'frozen_contract.json');check_binding=bind(prep/'scene_adapter_check.json');check=verified_json(check_binding)
    if check['status']!='pass_open_natural_data_adapter_rehearsal' or check['recording']!='04' or check['protected_data_read']:
        raise PermissionError('only the known open STOP04 software-test scenes can be used here')
    root=prep/'end_to_end_open_STOP04_v1';root.mkdir(exist_ok=False)
    db=finalize_dataset([check['result']],scope='STOP',expected_recordings=['04'],contract_binding=contract,
        output=root/'dataset.json',rehearsal=True)
    dataset=CompleteFinalDataset(db,scope='STOP',allowed_recordings=['04'])
    if len(dataset)!=3:raise ValueError('fixed three-scene software QA scope changed')
    references=evaluate_references(dataset,contract,root/'references',device='cpu')
    queue=prepare_queue(dataset,contract,root/'queue');results={}
    for arm in ('canonical','atom_aware'):results[arm]=run_generation(queue,contract,root/'generation',arm)
    audit=audit_generation(queue,contract,results,root/'generation_audit.json')
    analysis=describe_generation(queue,results,audit,root/'generation_analysis.json')
    numeric=verified_json(audit)
    if numeric['retained_failures']!=0 or numeric['replayed_complete']!=90:
        raise RuntimeError('software test requires all 90 finite trajectories; inspect recorded failures without resampling')
    report=dict(status='pass_open_data_end_to_end_software_rehearsal',contract=contract,source_check=check_binding,
        dataset=db,reference_evaluation=references,queue=queue,generation_results=results,generation_audit=audit,
        generation_analysis=analysis,complete_natural_histories=3,actual_generated_requests=90,
        final_validation_performance=False,protected_data_read=False,one_recording_no_generalization_CI=True,
        wall_seconds=time.monotonic()-start)
    output=write_once(root/'result.json',report)
    print(json.dumps(dict(status=report['status'],output=output,final_data_read=False)),flush=True)


if __name__=='__main__':run()
