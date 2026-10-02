"""Experimental P + CDF shape + exact estimated quantile FiLM adapter.

P remains a direct condition; target PET is computed only from the frozen
history-conditioned CDF and P, never from observed/generated future. This
candidate is NOT wired into production or the currently running shape trial.
"""
import math
from collections.abc import Mapping
import torch
from torch import nn
from .ruler_context_direct_p import RulerContextPercentileDenoiser
from .cdf_shape_context import CONTEXT_KEY
from .diffusion import FEATURE_KEYS,_clean_coefficients,_timesteps
from pcontrol.reference.torch_frozen_inverse import PIECES_KEY,quantile_from_pieces


class TargetRulerPercentileDenoiser(RulerContextPercentileDenoiser):
    VERSION='natural_P_CDF_shape_exact_target_quantile_adapter_v1'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.target_projection=nn.Linear(1,self.context_hidden_dim,bias=False)
        nn.init.zeros_(self.target_projection.weight)

    def load_from_shape_state_dict(self,state):
        expected=self.state_dict();key='target_projection.weight'
        if not isinstance(state,Mapping) or set(state)!=set(expected)-{key}:
            raise ValueError('complete shape-CFG state required')
        for name,value in state.items():
            if (not isinstance(value,torch.Tensor) or value.shape!=expected[name].shape or value.dtype!=expected[name].dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('invalid shape parent tensor: '+name)
        expanded=dict(state);expanded[key]=torch.zeros_like(expected[key])
        self.load_state_dict(expanded,strict=True)
        return dict(shape_parent_loaded=True,only_added_target_column_zero=True,learned_null_preserved=True)

    def load_from_parent_state_dict(self,state):
        raise ValueError('target extension requires an explicit complete shape-CFG parent')

    def train_adapter_only(self):
        self.requires_grad_(False)
        self.ruler_encoder.requires_grad_(True);self.target_projection.requires_grad_(True)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _ruler_features(self,features):
        if not isinstance(features,Mapping) or set(features)!=FEATURE_KEYS|{CONTEXT_KEY,PIECES_KEY}:
            raise ValueError('H/static, CDF shape, and exact frozen physical pieces required')
        base={k:v for k,v in features.items() if k!=PIECES_KEY}
        cleaned,shape=super()._ruler_features(base)
        # Validate H-only table even when p is dropped. Never accept future fields.
        quantile_from_pieces(features[PIECES_KEY],torch.full((len(shape),),.5,device=shape.device))
        return cleaned,shape

    def architecture_config(self):
        config=super().architecture_config()
        config.update(version=self.VERSION,feature_keys=sorted(FEATURE_KEYS|{CONTEXT_KEY,PIECES_KEY}),
            target_PET_input=True,target_PET_source='exact_frozen_H_CDF_inverse_of_1_minus_requested_P',
            actual_future_input=False,p_condition_is_actual_model_input=True,
            total_P_gradient='direct_P_embedding_plus_exact_inverse_CDF_query_path',
            CDF_shape_plus_target_input_dimension=66,
            target_injection='separate_zero_Linear1_to_hidden_added_to_unchanged_Linear65_output',
            target_quantile_normalization='divide_PET_seconds_by_fixed_cap4',
            unconditional_branch='P_and_both_adapter_cues_disabled_together',
            risk_oracle_future_calls=0,prototype_not_production=True)
        return config

    def forward(self,noisy_coefficients,timesteps,features,p,condition_present=None):
        # Mirror the authenticated parent arithmetic; no frozen source edits.
        (h,dims,road,mask,ego,road_mask),shape=self._ruler_features(features)
        noisy=_clean_coefficients(noisy_coefficients,mask,name='noisy coefficients')
        if (noisy.dtype != torch.float32 or noisy.device != h.device
                or noisy.shape[:2] != mask.shape or math.prod(noisy.shape[2:]) != self.coefficient_dim):
            raise ValueError('float32 noisy coefficients must match the parent denoiser')
        batch,count=mask.shape; t=_timesteps(timesteps,batch,noisy.device)
        temporal=h.permute(0,2,1,3).reshape(batch,count,52)
        observed=self.history_encoder(torch.cat((temporal,dims,ego[...,None].to(h.dtype)),-1))
        road_tokens=self.road_encoder(road[...,None])
        road_mean=torch.where(road_mask[...,None],road_tokens,torch.zeros_like(road_tokens)).sum(1)
        road_mean=road_mean / road_mask.sum(1).to(h.dtype)[:,None]
        road_max=road_tokens.masked_fill(~road_mask[...,None],-torch.inf).max(1).values
        context=self.road_pool_encoder(torch.cat((road_mean,road_max),-1))
        context=context+self.count_encoder(torch.log1p(mask.sum(1).to(h.dtype))[:,None])
        phase=t.to(h.dtype)[:,None]*self.time_frequencies[None]
        time=self.time_encoder(torch.cat((torch.sin(phase),torch.cos(phase)),-1))
        percentile,present=self._condition_embedding(p,condition_present,batch,noisy.device)
        safe_p=torch.where(present,p,torch.full_like(p,.5))
        target=quantile_from_pieces(features[PIECES_KEY],1.-safe_p.double())/4.
        hidden=self.ruler_encoder[0](shape)+self.target_projection(target.to(shape.dtype)[:,None])
        scale,shift=self.ruler_encoder[2](self.ruler_encoder[1](hidden)).chunk(2,-1)
        residual=scale*percentile+shift
        percentile=percentile+torch.where(present[:,None],residual,torch.zeros_like(residual))
        tokens=observed+self.coefficient_encoder(noisy.reshape(batch,count,self.coefficient_dim))
        tokens=tokens+context[:,None]+time[:,None]+percentile[:,None]
        for block in self.blocks:
            tokens=block(tokens,mask,percentile,time)
        prediction=self.output(tokens)
        prediction=torch.where(mask[...,None],prediction,torch.zeros_like(prediction))
        return prediction.reshape_as(noisy_coefficients)
