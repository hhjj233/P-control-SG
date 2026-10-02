"""Same P5 generator/controller as a data-source-independent final adapter.

No dataset, observed future, actor identity, training update or sample selection
enters the network interface. File-reading authorization belongs to the runner.
"""
import time
import numpy as np
import torch
from pcontrol.publication_pipeline.final_validation_protocol import verified_json,same_binding,digest,resolve
from pcontrol.publication_pipeline.final_reference_suite import FrozenFinalReference,FEATURES
from pcontrol.generation.atom_aware_rank_target import select_midrank_target
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.generation.trajectory_basis import TrajectoryBasis
from pcontrol.generation.risk_guidance import TorchTrajectoryDecoder
from pcontrol.generation.cdf_shape_context import shape_from_reference,CONTEXT_KEY
from pcontrol.reference.torch_frozen_inverse import PIECES_KEY
from pcontrol.generation.percentile_sampling_guidance import percentile_guided_sample
from pcontrol.generation.background_constrained_guidance import BackgroundConstrainedGuidance
from pcontrol.generation.evaluation import quality_metrics
from pcontrol.generation.scene_quality_diagnostics import scene_quality
from pcontrol.time_attention_pipeline import common as c
from pcontrol.research import train_guarded_terminal_pair as training
from pcontrol.research.evaluate_natural_diffusion import model_features
from pcontrol.research.refine_natural_direct_p import state_hash
from pcontrol.research.audit_pair_overlap_intervals import pair_overlap_intervals


def initialize_runtime():
    torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


class FrozenFinalGenerator:
    def __init__(self,contract_binding,arm):
        self.contract=verified_json(contract_binding)
        self.policy=verified_json(self.contract['policy'])
        if arm not in self.policy['generation']['arms']:raise ValueError('unknown paired arm')
        self.arm=arm
        previous=verified_json(self.contract['source_generation_freeze'])
        p5=verified_json(previous['policy']);self.training_policy=verified_json(p5['paired_training_policy'])
        c.verify_sources(previous['code_sha256'])
        binding=self.contract['generator_checkpoints'][arm]
        if not same_binding(binding,previous['checkpoints'][arm]) or digest(binding['path'])!=binding['sha256']:
            raise ValueError('frozen generator checkpoint drift')
        checkpoint=torch.load(resolve(binding['path']),map_location='cpu',weights_only=False)
        if (checkpoint['protocol']!=training.PROTOCOL or checkpoint['arm']!=arm or checkpoint['smoke']
                or checkpoint['epoch']!=3 or not same_binding(checkpoint['reference_manifest'],self.contract['main_reference'])):
            raise ValueError('wrong P5 generator state')
        with torch.random.fork_rng(devices=[]):
            self.model=training.model_from_checkpoint(checkpoint,arm,self.training_policy['target_policy'],torch.device('cpu')).eval().requires_grad_(False)
        if self.model.architecture_config()!=checkpoint['architecture']:raise ValueError('generator architecture drift')
        self.initial_state=state_hash(self.model)
        self.schedule=CosineDiffusionSchedule(100);self.basis=TrajectoryBasis(8)
        self.cn=verified_json(checkpoint['coefficient_normalizer']);self.hn=verified_json(checkpoint['history_normalizer'])
        self.plugin=FrozenFinalReference.from_contract(contract_binding,'M2_TimeAttn',device='cpu')
        self.profile=self.contract['generation_profile']
        if self.profile!=previous['profile']:raise ValueError('controller differs from frozen P5 profile')

    def prepare_case(self, features):
        if set(features)!=FEATURES:raise ValueError('generator accepts only explicit physical history/static features')
        reference=self.plugin.condition_features(features);n=reference.num_agents
        if n!=features['history'].shape[1] or not features['agent_mask'].all():
            raise ValueError('final case must preserve every actual actor without padded/noise slots')
        pieces=c.StableTorchInverse(reference._masses,[n],warp=reference._warp,base_knots=reference._knots).compiled_pieces()
        shape=np.asarray(shape_from_reference(reference),dtype=np.float32)
        case=dict(features,num_agents=n,future_dt=.04)
        tensors=model_features(case,self.hn,torch.device('cpu'))
        tensors[CONTEXT_KEY]=torch.tensor(shape[None]);tensors[PIECES_KEY]=pieces
        decoder=TorchTrajectoryDecoder(self.basis,self.cn,case['history'][-1])
        return dict(case=case,reference=reference,pieces=pieces,features=tensors,decoder=decoder,
            ego=int(np.flatnonzero(case['ego_mask'])[0]))

    def sample(self, prepared, requested, noise):
        if requested not in self.policy['generation']['P_grid']:raise ValueError('undeclared requested percentile')
        case=prepared['case'];reference=prepared['reference'];n=case['num_agents']
        if noise.shape!=(n,8,2) or noise.dtype!=np.float32 or not np.isfinite(noise).all():
            raise ValueError('finite full-roster float32 noise required')
        scalar=select_midrank_target(prepared['pieces'][0].numpy(),requested,**self.training_policy['target_policy'])
        target=scalar['canonical_quantile_pet_seconds'] if self.arm=='canonical' else scalar['control_target_pet_seconds']
        guide=BackgroundConstrainedGuidance(prepared['decoder'],case['dimensions'],case['history'][-1],
            lambda y:float(reference.rank(y)['p_mid']),requested,target,ego_index=prepared['ego'],
            road_boundaries=case['road_boundaries'],**self.profile)
        start=time.monotonic()
        info=percentile_guided_sample(self.model,self.schedule,prepared['features'],
            torch.tensor([requested],dtype=torch.float32),torch.tensor(noise[None]),guide,scale=2.5)
        coefficients=info.pop('sample')[0].numpy().astype(np.float64)
        future=self.basis.decode(coefficients*np.asarray(self.cn['scale'])+np.asarray(self.cn['mean']),case['history'][-1])
        elapsed=time.monotonic()-start
        if not np.isfinite(future).all() or not np.array_equal(future[0],case['history'][-1]):
            raise FloatingPointError('invalid generated future or changed initial state')
        measured=reference.score_future(future);pet=float(measured['pet_seconds']);rank=measured['estimated_rank']
        overlaps=pair_overlap_intervals(future,case['dimensions'],prepared['ego'])
        row=dict(status='complete',arm=self.arm,requested_p=requested,pet_seconds=pet,estimated_rank=rank,
            p_mid_absolute_error=abs(rank['p_mid']-requested),p_interval_error=max(rank['p_low']-requested,requested-rank['p_up'],0.),
            control_target_PET_seconds=target,canonical_target_PET_seconds=scalar['canonical_quantile_pet_seconds'],
            PET_control_target_absolute_error_seconds=abs(pet-target),
            canonical_PET_target_absolute_error_seconds=abs(pet-scalar['canonical_quantile_pet_seconds']),
            scalar_error_infimum=scalar['unrestricted_scalar_error_infimum'],atom_target_diagnostics=scalar,
            scene_quality=scene_quality(future,case['dimensions'],case['ego_mask']),
            PL_overlap_intervals=overlaps,background_PL_overlap_scene=any(x['background_pair'] for x in overlaps),
            ego_PL_overlap_scene=any(not x['background_pair'] for x in overlaps),guidance=info['guidance'],
            network_evaluations=info['network_evaluations'],sampling_seconds=elapsed,K=1,CFG_scale=2.5,
            all_actors_retained=True,post_sampler_repair=False,inference_observed_future_input=False)
        return future,row

    def assert_unchanged(self):
        if state_hash(self.model)!=self.initial_state or any(p.grad is not None or p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('frozen generation changed model state')


def attach_observation_diagnostics(row,future,features,observed_future):
    """Only after sampling: recorded future is descriptive context, not input."""
    case=dict(features,num_agents=features['history'].shape[1],future_dt=.04,future=observed_future)
    row=dict(row,quality=quality_metrics(future,case),observed_future_used_only_after_sampling=True)
    return row
