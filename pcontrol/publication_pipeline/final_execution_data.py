"""Dataset/queue adapters shared by gated final execution and open rehearsals."""
import json
from pathlib import Path
import numpy as np
from pcontrol.publication_pipeline.final_validation_protocol import (
    bind,verified_json,resolve,same_binding,select_histories,noise_for_history)
from pcontrol.publication_pipeline.final_scene_data import write_once,DATA_PROTOCOL
from pcontrol.publication_pipeline.final_reference_suite import FrozenFinalReference,FEATURES
from pcontrol.data.complete_scene_view import complete_eligibility,validate_complete_labels,reference_example
from pcontrol.generation.cdf_shape_context import CONTEXT_KEY,shape_from_reference
from pcontrol.reference.torch_frozen_inverse import PIECES_KEY
from pcontrol.time_attention_pipeline import common as c


def save_arrays(path,arrays):
    with Path(path).open('xb') as handle:np.savez_compressed(handle,**arrays)
    return bind(path)


def finalize_dataset(recording_results, *, scope, expected_recordings, contract_binding, output, rehearsal=False):
    if scope=='STOP' and not rehearsal:raise PermissionError('STOP is software rehearsal, never a final cohort')
    contract=verified_json(contract_binding);policy=verified_json(contract['policy'])
    allowed=verified_json(policy['sources']['development_roles'])['STOP'] if rehearsal else policy['cohorts'][scope]
    if (not set(expected_recordings)<=set(allowed) or len(expected_recordings)!=len(set(expected_recordings))
            or (not rehearsal and list(expected_recordings)!=list(allowed))):
        raise PermissionError('dataset recording roster not authorized by its protocol')
    records={};ids=[]
    for binding in recording_results:
        item=verified_json(binding);rec=item['recording_id']
        if item['protocol']!=DATA_PROTOCOL or item['status']!='complete' or item['role']!=scope:
            raise ValueError('wrong complete recording result')
        if rec not in expected_recordings or rec in records:raise ValueError('missing/duplicate/foreign recording')
        if not same_binding(item['provenance']['contract'],contract_binding):raise ValueError('dataset model contract drift')
        if item['selected_rows_replaced'] or not item['all_incomplete_candidates_retained_in_ledger']:
            raise ValueError('incomplete rows were replaced or hidden')
        if len(item['complete_scene_ids'])!=item['counts'].get('complete',0):raise ValueError('complete count mismatch')
        records[rec]=dict(result=binding,data=item['data'],scene_ids=item['complete_scene_ids'],
            num_agents=item['num_agents'],counts=item['counts'])
        ids.extend(item['complete_scene_ids'])
    if set(records)!=set(expected_recordings) or len(ids)!=len(set(ids)):raise ValueError('incomplete or duplicate dataset roster')
    report=dict(protocol='final_complete_scene_dataset_v1',status='complete',scope=scope,
        contract=contract_binding,recordings=records,complete_scenes=len(ids),expected_recordings=list(expected_recordings),
        software_rehearsal=rehearsal,recording_disjoint_final_data=not rehearsal,
        never_used_before_claim=False,natural_only=True,no_incomplete_replacement=True)
    return write_once(output,report)


class CompleteFinalDataset:
    """Caller must pass its approved role/recording roster before any NPZ read."""
    def __init__(self,binding,*,scope,allowed_recordings):
        self.binding=binding;self.manifest=verified_json(binding)
        if (self.manifest['protocol']!='final_complete_scene_dataset_v1' or self.manifest['scope']!=scope
                or not set(self.manifest['recordings'])<=set(allowed_recordings)):
            raise PermissionError('dataset outside supplied evaluation scope')
        self.rows=[]
        for rec,entry in sorted(self.manifest['recordings'].items()):
            if len(entry['scene_ids'])!=len(entry['num_agents']):raise ValueError('identity/count mismatch')
            self.rows.extend(dict(scene_id=sid,recording_id=rec,num_agents=int(n),role=scope,
                source_row=i,source_shard=entry['data']) for i,(sid,n) in enumerate(zip(entry['scene_ids'],entry['num_agents'])))
        if len(self.rows)!=self.manifest['complete_scenes']:raise ValueError('dataset denominator mismatch')
        self._loaded=None;self._arrays=None

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];binding=row['source_shard']
        if self._loaded!=binding:
            self._arrays=c.arrays(binding);self._loaded=binding
            complete=complete_eligibility(self._arrays);validate_complete_labels(self._arrays,complete)
            if not complete.all():raise ValueError('incomplete scene in final point-label population')
        arrays=self._arrays;i=row['source_row']
        if (str(arrays['scene_id'][i])!=row['scene_id'] or str(arrays['recording_id'][i])!=row['recording_id']
                or str(arrays['role'][i])!=row['role']):raise ValueError('actual shard identity differs from declared metadata')
        example=reference_example(arrays,i);lo,hi=map(int,arrays['offsets'][i:i+2]);n=hi-lo
        if n!=row['num_agents']:raise ValueError('roster count drift')
        features=example['features'];features['agent_mask']=np.ones(n,bool)
        future=arrays['future_native_agents'][lo:hi].transpose(1,0,2).copy()
        if not np.array_equal(future[0],features['history'][-1]):raise ValueError('recorded initial state changed')
        return dict(features=features,target=example['target'],future_observed=future,
            agent_ids=arrays['agent_ids'][lo:hi].copy(),metadata=row)


def prepare_queue(dataset,contract_binding,output_dir):
    contract=verified_json(contract_binding);p=verified_json(contract['policy']);g=p['generation']
    role=dataset.manifest['scope'];allowed=dataset.manifest['expected_recordings']
    identities=[{k:r[k] for k in ('scene_id','recording_id','num_agents','role')} for r in dataset.rows]
    chosen,selection=select_histories(identities,allowed_recordings=allowed,role=role,
        per_group=g['history_count_per_stratum_per_cohort'],salt=g['selection_salt'])
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=False)
    barrier=write_once(root/'identities_before_reference_queries.json',dict(contract=contract_binding,dataset=dataset.binding,
        selected=chosen,selection=selection,used_PET_or_model_predictions=False))
    lookup={r['scene_id']:i for i,r in enumerate(dataset.rows)}
    plugin=FrozenFinalReference.from_contract(contract_binding,'M2_TimeAttn',device='cpu');cases=[]
    for identity in chosen:
        item=dataset[lookup[identity['scene_id']]];features=item['features'];ref=plugin.condition_features(features)
        n=identity['num_agents'];pieces=c.StableTorchInverse(ref._masses,[n],warp=ref._warp,base_knots=ref._knots).compiled_pieces()[0].numpy()
        arrays=dict(features,agent_ids=item['agent_ids'],future_observed=item['future_observed'],natural_PET=np.asarray(item['target']),
            reference_joint_masses=ref._masses[0],reference_row_nodes=ref._warp.row_nodes([n])[0],
            **{CONTEXT_KEY:np.asarray(shape_from_reference(ref),np.float32),PIECES_KEY:pieces})
        noises={}
        for z in g['noise_indices']:
            arrays[f'initial_noise_{z}'],noises[str(z)]=noise_for_history(identity['scene_id'],z,n,salt=g['noise_salt'])
        artifact=save_arrays(root/f'case_{identity["case_index"]:03d}.npz',arrays)
        cases.append(dict(identity,artifact=artifact,noise=noises,old12=False))
    report=dict(protocol='final_generation_queue_v1',status='prepared',contract=contract_binding,dataset=dataset.binding,
        identities=barrier,scope=role,cases=cases,selection=selection,histories=len(cases),
        requests_per_arm=len(cases)*len(g['P_grid'])*len(g['noise_indices']),
        software_rehearsal=dataset.manifest['software_rehearsal'],all_failures_and_noises_required=True)
    return write_once(root/'queue.json',report)
