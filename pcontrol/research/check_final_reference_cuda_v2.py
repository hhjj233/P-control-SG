#!/usr/bin/env python3
"""Check the declared CUDA reference runtime on the same open rehearsal cohort."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.publication_pipeline.final_validation_protocol import bind,verified_json
from pcontrol.publication_pipeline.final_execution_data import CompleteFinalDataset
from pcontrol.publication_pipeline.final_execution_reference import evaluate_references
from pcontrol.publication_pipeline.final_generation_adapter import initialize_runtime
from pcontrol.publication_pipeline.final_scene_data import write_once
from pcontrol.time_attention_pipeline import common as c


def run():
    initialize_runtime();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('declared final reference device needs CUDA')
    prep=ROOT/'outputs/natural_percentile/transformer_publication_v1_20260916/final_validation_v1_preparation'
    source=verified_json(bind(prep/'end_to_end_open_STOP04_v1/result.json'))
    if source['protected_data_read'] or source['final_validation_performance']:raise PermissionError('open software cohort only')
    dataset=CompleteFinalDataset(source['dataset'],scope='STOP',allowed_recordings=['04'])
    output=evaluate_references(dataset,source['contract'],prep/'reference_cuda_open_STOP04_v2',device='cuda:0')
    gpu=verified_json(output);cpu=verified_json(source['reference_evaluation']);maximum=0.
    for arm in gpu['models']:
        for view in ('raw','frozen_calibrated'):
            a=c.arrays(gpu['models'][arm]['predictions'][view]);b=c.arrays(cpu['models'][arm]['predictions'][view])
            for key in ('scene_id','recording_id','num_agents','target'):
                if not np.array_equal(a[key],b[key]):raise ValueError('CUDA test cohort changed')
            maximum=max(maximum,float(np.max(abs(a['joint_masses']-b['joint_masses']))))
    if maximum>2e-6:raise RuntimeError('unexpected CPU/CUDA probability discrepancy: '+str(maximum))
    result=dict(status='pass_declared_CUDA_reference_runtime',source_rehearsal=bind(prep/'end_to_end_open_STOP04_v1/result.json'),
        reference_result=output,maximum_CPU_CUDA_probability_difference=maximum,acceptance_tolerance=2e-6,
        independent_score_replay_passed=True,protected_data_read=False,not_new_generalization_evidence=True,
        GRU_cuda_backend='native_PyTorch_no_cudnn_fusion',old_failed_CUDA_run_preserved=True,
        device=torch.cuda.get_device_name(0),code=bind(__file__))
    out=write_once(prep/'reference_cuda_check_v2.json',result)
    print(json.dumps(dict(status=result['status'],maximum_difference=maximum,result=out)),flush=True)


if __name__=='__main__':run()
