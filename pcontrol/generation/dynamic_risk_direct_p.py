"""Candidate dynamic actor-token risk adapters on the frozen joint denoiser.

Unlike a scene-global static condition shift, each added branch reads the
current denoising tokens (history/noisy future/interactions already encoded by
the parent). It can adapt different real actors differently. No future PET
oracle, gradient guidance, output correction, or candidate selection is used.
Zero-output branches initially preserve a complete trained target-ruler parent.
"""
import math
from collections.abc import Mapping
import torch
from torch import nn
from .target_ruler_direct_p import TargetRulerPercentileDenoiser
from .diffusion import _clean_coefficients,_timesteps
from pcontrol.reference.torch_frozen_inverse import PIECES_KEY,quantile_from_pieces


class DynamicRiskResidual(nn.Module):
    def __init__(self,width,bottleneck):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.token_down=nn.Linear(width,bottleneck)
        self.risk_down=nn.Linear(width,bottleneck)
        self.output=nn.Linear(bottleneck,width)
        self.zero_output()

    def zero_output(self):
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,tokens,risk,mask,present):
        hidden=torch.nn.functional.silu(self.token_down(self.norm(tokens))+self.risk_down(risk)[:,None])
        value=self.output(hidden)
        return torch.where((mask & present[:,None])[...,None],value,torch.zeros_like(value))


class DynamicRiskPercentileDenoiser(TargetRulerPercentileDenoiser):
    VERSION='natural_P_CDF_exact_target_dynamic_actor_risk_adapter_v1'

    def __init__(self,*args,dynamic_bottleneck=32,**kwargs):
        super().__init__(*args,**kwargs)
        if type(dynamic_bottleneck) is not int or dynamic_bottleneck<1:raise ValueError('positive adapter width required')
        self.dynamic_bottleneck=dynamic_bottleneck
        self.dynamic_blocks=nn.ModuleList([DynamicRiskResidual(self.hidden_dim,dynamic_bottleneck) for _ in self.blocks])

    def load_from_target_state_dict(self,state):
        expected=self.state_dict();new={k for k in expected if k.startswith('dynamic_blocks.')}
        if not isinstance(state,Mapping) or set(state)!=set(expected)-new:
            raise ValueError('complete trained target-ruler parent required')
        for name,value in state.items():
            if (not isinstance(value,torch.Tensor) or value.shape!=expected[name].shape or value.dtype!=expected[name].dtype
                    or not bool(torch.isfinite(value).all())):raise ValueError('invalid parent tensor: '+name)
        missing=self.load_state_dict(state,strict=False)
        if set(missing.missing_keys)!=new or missing.unexpected_keys:raise RuntimeError('unexpected warm-start keys')
        for block in self.dynamic_blocks:block.zero_output()
        return dict(complete_target_parent_loaded=True,new_dynamic_outputs_zeroed=True,learned_null_preserved=True)

    def load_from_shape_state_dict(self,state):
        raise ValueError('use an explicitly bound complete target-ruler parent')

    def train_adapter_only(self):
        super().train_adapter_only()
        self.dynamic_blocks.requires_grad_(True)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_config(self):
        config=super().architecture_config()
        config.update(version=self.VERSION,dynamic_bottleneck=self.dynamic_bottleneck,
            dynamic_adapter_scope='per_actor_after_each_existing_joint_attention_FiLM_block',
            dynamic_adapter_inputs='current_denoising_tokens_and_scene_P_CDF_target_embedding',
            dynamic_adapter_disabled_on_null_P_and_padding=True,
            dynamic_adapter_zero_output_initialization=True,
            external_future_risk_oracle=False,prototype_not_production=True,
            EMA_update_scope_required='ruler_encoder,target_projection,dynamic_blocks ONLY')
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
        for block,adapter in zip(self.blocks,self.dynamic_blocks):
            tokens=block(tokens,mask,percentile,time)
            tokens=tokens+adapter(tokens,percentile,mask,present)
        prediction=self.output(tokens)
        prediction=torch.where(mask[...,None],prediction,torch.zeros_like(prediction))
        return prediction.reshape_as(noisy_coefficients)

