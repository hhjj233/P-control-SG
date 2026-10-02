"""Experimental P + frozen H-only CDF-shape FiLM adapter, not production.

The public input contract is deliberately extended by one CDF-shape tensor.
It is built once per history outside the sampler. No generated-future scoring,
target-PET input, inference gradient, projection, or best-of-K exists here.
Zero-output initialization exactly replays the trained CFG parent, including
its learned null embedding. The adapter is masked out when p is dropped.
"""
import math
from collections.abc import Mapping

import torch
from torch import nn

from .cdf_shape_context import CONTEXT_DIM, CONTEXT_KEY, descriptor_contract
from .direct_p_cfg import ClassifierFreePercentileDenoiser
from .diffusion import FEATURE_KEYS, _clean_coefficients, _timesteps


class RulerContextPercentileDenoiser(ClassifierFreePercentileDenoiser):
    VERSION = 'natural_P_plus_H_only_CDF_shape_zero_FiLM_adapter_v1'

    def __init__(self, coefficient_dim, hidden_dim=128, heads=4, layers=3,
                 feedforward_dim=256, context_hidden_dim=64):
        super().__init__(coefficient_dim, hidden_dim, heads, layers, feedforward_dim)
        if type(context_hidden_dim) is not int or context_hidden_dim < 1:
            raise ValueError('positive integer context width required')
        self.context_hidden_dim = context_hidden_dim
        self.ruler_encoder = nn.Sequential(nn.Linear(CONTEXT_DIM,context_hidden_dim),nn.SiLU(),
                                           nn.Linear(context_hidden_dim,2*hidden_dim))
        self.zero_adapter_output()

    def zero_adapter_output(self):
        nn.init.zeros_(self.ruler_encoder[-1].weight)
        nn.init.zeros_(self.ruler_encoder[-1].bias)

    def load_from_parent_state_dict(self, state):
        expected=self.state_dict(); new={k for k in expected if k.startswith('ruler_encoder.')}
        if not isinstance(state,Mapping) or set(state) != set(expected)-new:
            raise ValueError('complete trained CFG parent, including learned null, required')
        for name,value in state.items():
            if (not isinstance(value,torch.Tensor) or value.shape != expected[name].shape
                    or value.dtype != expected[name].dtype or not bool(torch.isfinite(value).all())):
                raise ValueError('invalid parent tensor: '+name)
        missing=super().load_state_dict(state,strict=False)
        if set(missing.missing_keys) != new or missing.unexpected_keys:
            raise RuntimeError('unexpected warm-start keys')
        self.zero_adapter_output()
        return dict(parent_loaded=True,learned_null_retained=True,new_output_zeroed=True,
                    new_tensor_keys=sorted(new))

    def load_from_direct_p_state_dict(self, state):
        raise ValueError('use a trained CFG parent; do not recreate its learned null')

    def train_adapter_only(self):
        self.requires_grad_(False)
        self.ruler_encoder.requires_grad_(True)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_config(self):
        config=super().architecture_config()
        config.update(version=self.VERSION,context_hidden_dim=self.context_hidden_dim,
            feature_keys=sorted(FEATURE_KEYS | {CONTEXT_KEY}),cdf_context=descriptor_contract(),
            weight_provenance='external_trained_CFG_parent_or_strict_adapter_checkpoint',
            risk_estimator_dependency=True,external_risk_estimator_or_oracle=True,
            estimator_dependency_scope='one_H_only_CDF_context_computed_outside_sampler; zero_future_oracle_calls',
            classifier_free_p_only=False,request_dropout='p_and_its_ruler_adapter_absent_together; all_H_static_retained',
            history_or_static_dropout=False,cdf_adapter_active_only_if_p_present=True,
            null_initialization='preserve_learned_CFG_parent_null_without_reinitialization',
            context_injection='percentile_embedding + shape_scale*percentile_embedding + shape_shift',
            new_output_zero_initialized=True,prototype_not_production=True)
        return config

    def _ruler_features(self, features):
        if not isinstance(features,Mapping) or set(features) != FEATURE_KEYS | {CONTEXT_KEY}:
            raise ValueError('six H/static fields plus explicit frozen_cdf_shape required')
        base={k:features[k] for k in FEATURE_KEYS}
        cleaned=super()._clean_features(base)
        h=cleaned[0]; shape=features[CONTEXT_KEY]
        if (not isinstance(shape,torch.Tensor) or shape.shape != (h.shape[0],CONTEXT_DIM)
                or shape.dtype != torch.float32 or shape.device != h.device or shape.requires_grad
                or not bool(torch.isfinite(shape).all()) or bool(((shape < 0) | (shape > 1)).any())
                or bool((torch.diff(shape,dim=-1) < -1e-6).any())):
            raise ValueError('frozen finite monotone float32[B,65] CDF shape required')
        return cleaned,shape

    def _clean_features(self, features):
        # The existing suffix-training helper validates through this method.
        return self._ruler_features(features)[0]

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
        scale,shift=self.ruler_encoder(shape).chunk(2,-1)
        residual=scale*percentile+shift
        percentile=percentile+torch.where(present[:,None],residual,torch.zeros_like(residual))
        tokens=observed+self.coefficient_encoder(noisy.reshape(batch,count,self.coefficient_dim))
        tokens=tokens+context[:,None]+time[:,None]+percentile[:,None]
        for block in self.blocks:
            tokens=block(tokens,mask,percentile,time)
        prediction=self.output(tokens)
        prediction=torch.where(mask[...,None],prediction,torch.zeros_like(prediction))
        return prediction.reshape_as(noisy_coefficients)
