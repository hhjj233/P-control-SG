"""Same trainable neural generator, with the honest atom-aware target query.

No extra trainable layers; H/CDF/P inputs and the learned-null mechanism stay.
Only the scalar target fed to the existing target projection is changed.
"""
import math

import torch

from .trainable_risk_generator import TrainableRiskGenerator
from .diffusion import _clean_coefficients,_timesteps
from .torch_atom_aware_target import midrank_target_from_pieces
from pcontrol.reference.torch_frozen_inverse import PIECES_KEY


class AtomAwareRiskGenerator(TrainableRiskGenerator):
    VERSION='natural_atom_aware_scalar_target_full_generator_v1'

    def __init__(self,*args,inward_margin_seconds=.001,excess_rank_tolerance=1e-5,**kwargs):
        super().__init__(*args,**kwargs)
        if not 0<inward_margin_seconds<=.1 or not 0<excess_rank_tolerance<=1e-3:raise ValueError('valid fixed target tolerances required')
        self.inward_margin_seconds=float(inward_margin_seconds);self.excess_rank_tolerance=float(excess_rank_tolerance)

    def architecture_config(self):
        config=super().architecture_config()
        config.update(version=self.VERSION,
            target_PET_source='nearest_midrank_scalar_target_from_frozen_H_CDF_and_original_P',
            total_P_gradient='direct_P_embedding_plus_selected_continuous_target_branch',
            target_quantile_normalization='divide_control_PET_seconds_by_fixed_cap4',
            target_policy=dict(inward_margin_seconds=self.inward_margin_seconds,excess_rank_tolerance=self.excess_rank_tolerance),
            requested_P_relabelled=False,scalar_target_physical_reachability_guaranteed=False,
            model_architecture_operators_unchanged=False,trainable_neural_architecture_unchanged=True,
            higher_order_autograd_requires_math_attention=True,
            null_initialization='constructor_p_encoder_at_0.5; overwritten_if_loading_complete_checkpoint',
            weight_provenance='explicit_checkpoint_header; fresh_initialization_or_authenticated_warm_start')
        config.pop('learned_null_loaded_without_reset',None)
        return config

    def forward(self,noisy_coefficients,timesteps,features,p,condition_present=None):
        # Mirror the immutable DynamicRiskPercentileDenoiser arithmetic except
        # for this version's explicit target query. Never monkeypatch a module.
        (h,dims,road,mask,ego,road_mask),shape=self._ruler_features(features)
        noisy=_clean_coefficients(noisy_coefficients,mask,name='noisy coefficients')
        if (noisy.dtype!=torch.float32 or noisy.device!=h.device or noisy.shape[:2]!=mask.shape
                or math.prod(noisy.shape[2:])!=self.coefficient_dim):raise ValueError('compatible float32 masked coefficients required')
        batch,count=mask.shape;t=_timesteps(timesteps,batch,noisy.device)
        temporal=h.permute(0,2,1,3).reshape(batch,count,52)
        observed=self.history_encoder(torch.cat((temporal,dims,ego[...,None].to(h.dtype)),-1))
        road_tokens=self.road_encoder(road[...,None])
        road_mean=torch.where(road_mask[...,None],road_tokens,torch.zeros_like(road_tokens)).sum(1)/road_mask.sum(1).to(h.dtype)[:,None]
        road_max=road_tokens.masked_fill(~road_mask[...,None],-torch.inf).max(1).values
        context=self.road_pool_encoder(torch.cat((road_mean,road_max),-1))+self.count_encoder(torch.log1p(mask.sum(1).to(h.dtype))[:,None])
        phase=t.to(h.dtype)[:,None]*self.time_frequencies[None]
        time=self.time_encoder(torch.cat((torch.sin(phase),torch.cos(phase)),-1))
        percentile,present=self._condition_embedding(p,condition_present,batch,noisy.device)
        safe_p=torch.where(present,p,torch.full_like(p,.5))
        target=midrank_target_from_pieces(features[PIECES_KEY],safe_p,
            inward_margin_seconds=self.inward_margin_seconds,excess_rank_tolerance=self.excess_rank_tolerance)/4.
        hidden=self.ruler_encoder[0](shape)+self.target_projection(target.to(shape.dtype)[:,None])
        scale,shift=self.ruler_encoder[2](self.ruler_encoder[1](hidden)).chunk(2,-1)
        percentile=percentile+torch.where(present[:,None],scale*percentile+shift,torch.zeros_like(percentile))
        tokens=observed+self.coefficient_encoder(noisy.reshape(batch,count,self.coefficient_dim))
        tokens=tokens+context[:,None]+time[:,None]+percentile[:,None]
        for block,adapter in zip(self.blocks,self.dynamic_blocks):
            tokens=block(tokens,mask,percentile,time);tokens=tokens+adapter(tokens,percentile,mask,present)
        prediction=self.output(tokens);prediction=torch.where(mask[...,None],prediction,torch.zeros_like(prediction))
        return prediction.reshape_as(noisy_coefficients)
