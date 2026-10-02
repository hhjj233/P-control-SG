"""Classifier-free guidance of requested p, while keeping all history/static H.

The only droppable condition is p. A learned scene-level null-p embedding
replaces p at the input AND every FiLM block. Training receives an explicit
boolean presence mask; this module never draws its own dropout mask. The
sampler combines the same network's v predictions and uses one initial noise
tensor. No PET target, reference estimator, oracle, correction or candidate
selection is imported or used.
"""
import math
from collections.abc import Mapping

import torch
from torch import nn

from .direct_p import JointPercentileDenoiser
from .diffusion import _clean_coefficients, _timesteps, diffusion_training_loss, ddim_sample


def _presence(condition_present, batch, device):
    if condition_present is None:
        return torch.ones(batch, dtype=torch.bool, device=device)
    if (not isinstance(condition_present, torch.Tensor) or condition_present.shape != (batch,)
            or condition_present.dtype != torch.bool or condition_present.device != device):
        raise ValueError('condition_present must be device-matched bool[B]; only p can be absent')
    return condition_present


class ClassifierFreePercentileDenoiser(JointPercentileDenoiser):
    """Direct-P architecture plus one learnable null_p_embedding[hidden_dim].

    forward(x,t,features,p,condition_present=None) keeps all six H/static inputs.
    None means all p present. An absent p is sanitized before its MLP, so even
    NaN/Inf placeholder values on absent rows have no output or gradient effect.
    Present p values still must be finite in[0,1]. No numerical zero p is used
    as an implicit absence flag: p=0 is a genuine conditional request.
    """
    VERSION = 'natural_direct_p_only_classifier_free_FiLM_denoiser_v1'

    def __init__(self, coefficient_dim, hidden_dim=128, heads=4, layers=3,
                 feedforward_dim=256):
        super().__init__(coefficient_dim, hidden_dim=hidden_dim, heads=heads,
                         layers=layers, feedforward_dim=feedforward_dim)
        with torch.no_grad():
            midpoint = self.percentile_embedding(torch.tensor([.5], dtype=torch.float32))[0]
        self.null_p_embedding = nn.Parameter(midpoint.detach().clone())

    def load_from_direct_p_state_dict(self, state_dict):
        """Strict parent-GP warm start; the ONLY new tensor is the null embedding.

        Parent tensors must have exactly the expected keys, shapes and dtypes.
        After loading them, initialize null from the loaded parent p_encoder(.5),
        not from the earlier random constructor state. Ordinary CFG checkpoint
        reloads should instead use inherited load_state_dict(..., strict=True).
        """
        expected = self.state_dict()
        names = set(expected) - {'null_p_embedding'}
        if not isinstance(state_dict, Mapping) or set(state_dict) != names:
            raise ValueError('parent GP must supply every and only pre-null architecture tensor')
        for name in names:
            value = state_dict[name]
            if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                    or value.dtype != expected[name].dtype or not bool(torch.isfinite(value).all())):
                raise ValueError('parent GP tensor shape/dtype/finite contract failed: ' + name)
        incompatible = super().load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys != ['null_p_embedding'] or incompatible.unexpected_keys:
            raise RuntimeError('only the declared null parameter may be missing from a GP warm start')
        with torch.no_grad():
            p = torch.tensor([.5], dtype=torch.float32, device=self.p_frequencies.device)
            self.null_p_embedding.copy_(self.percentile_embedding(p)[0])
        return dict(parent_state_loaded=True, only_new_parameter='null_p_embedding',
                    null_initialized_from_loaded_parent_p_encoder_at=.5,
                    conditional_parent_replay_requires_all_present=True)

    def architecture_config(self):
        config = super().architecture_config()
        # The new experiments deliberately warm-start trained natural GP
        # weights. Do not inherit the old fresh-model provenance assertion.
        config.pop('pretrained_weights_loaded', None)
        config.update(version=self.VERSION, classifier_free_p_only=True,
            null_p_embedding_shape=[self.hidden_dim], null_p_embedding_learnable=True,
            null_initialization='loaded_parent_p_encoder_at_0.5',
            history_or_static_dropout=False, default_condition_present=True,
            training_presence_mask='external_explicit_bool_B_no_internal_RNG',
            absence_is_not_numeric_p_zero=True,
            null_injection='same_input_and_every_FiLM_location_as_present_p',
            weight_provenance='external_parent_checkpoint_binding_or_CFG_checkpoint_reload',
            sampling_combination='v_null+scale*(v_cond-v_null); exact_single_branch_at_scale_0_or_1',
            additional_sampling_noise=False, external_risk_estimator_or_oracle=False)
        return config

    def _condition_embedding(self, p, condition_present, batch, device):
        mask = _presence(condition_present, batch, device)
        if (not isinstance(p, torch.Tensor) or p.shape != (batch,)
                or p.device != device or p.dtype not in (torch.float32, torch.float64)):
            raise ValueError('p must retain floating [B] shape/device, including absent rows')
        safe_p = torch.where(mask, p, torch.full_like(p, .5))
        present = self.percentile_embedding(safe_p)
        return torch.where(mask[:, None], present, self.null_p_embedding[None]), mask

    def forward(self, noisy_coefficients, timesteps, features, p, condition_present=None):
        # Preserve the frozen GP arithmetic order for exact all-present replay.
        h, dims, road, mask, ego, road_mask = self._clean_features(features)
        noisy = _clean_coefficients(noisy_coefficients, mask, name='noisy coefficients')
        if (noisy.dtype != torch.float32 or noisy.device != h.device
                or noisy.shape[:2] != mask.shape or math.prod(noisy.shape[2:]) != self.coefficient_dim):
            raise ValueError('float32 noisy coefficients must match the p-CFG denoiser')
        batch, count = mask.shape
        t = _timesteps(timesteps, batch, noisy.device)
        temporal = h.permute(0, 2, 1, 3).reshape(batch, count, 52)
        observed = self.history_encoder(torch.cat((temporal, dims, ego[..., None].to(h.dtype)), -1))
        road_tokens = self.road_encoder(road[..., None])
        road_mean = torch.where(road_mask[..., None], road_tokens, torch.zeros_like(road_tokens)).sum(1)
        road_mean = road_mean / road_mask.sum(1).to(h.dtype)[:, None]
        road_max = road_tokens.masked_fill(~road_mask[..., None], -torch.inf).max(1).values
        context = self.road_pool_encoder(torch.cat((road_mean, road_max), -1))
        context = context + self.count_encoder(torch.log1p(mask.sum(1).to(h.dtype))[:, None])
        phase = t.to(h.dtype)[:, None] * self.time_frequencies[None]
        time = self.time_encoder(torch.cat((torch.sin(phase), torch.cos(phase)), -1))
        percentile, _ = self._condition_embedding(p, condition_present, batch, noisy.device)
        tokens = observed + self.coefficient_encoder(noisy.reshape(batch, count, self.coefficient_dim))
        tokens = tokens + context[:, None] + time[:, None] + percentile[:, None]
        for block in self.blocks:
            tokens = block(tokens, mask, percentile, time)
        prediction = self.output(tokens)
        prediction = torch.where(mask[..., None], prediction, torch.zeros_like(prediction))
        return prediction.reshape_as(noisy_coefficients)


class _BoundTrainingCondition(nn.Module):
    def __init__(self, model, features, p, condition_present):
        super().__init__()
        self.model, self.features, self.p = model, features, p
        self.condition_present = condition_present

    def forward(self, noisy, timesteps, features):
        if features is not self.features:
            raise ValueError('p-CFG training context differs from its bound H/static features')
        return self.model(noisy, timesteps, features, self.p, self.condition_present)


def cfg_training_loss(model, schedule, x0, features, p, condition_present, *,
                      timesteps=None, noise=None, generator=None, prediction_type='v'):
    """Standard equal-scene v-MSE; explicit p-presence mask, no dropout RNG."""
    if not isinstance(model, ClassifierFreePercentileDenoiser) or prediction_type != 'v':
        raise ValueError('p-CFG training requires the declared model and v prediction')
    if condition_present is None:
        raise ValueError('training requires an explicit bool condition_present mask')
    _presence(condition_present, x0.shape[0], x0.device)
    adapter = _BoundTrainingCondition(model, features, p, condition_present)
    result = diffusion_training_loss(adapter, schedule, x0, features, timesteps=timesteps,
                                     noise=noise, generator=generator, prediction_type='v')
    result.update(conditioning_p=p, condition_present=condition_present,
                  presence_mask_generated_in_backend=False)
    return result


class _CFGPrediction(nn.Module):
    def __init__(self, model, features, p, scale):
        super().__init__()
        self.model, self.features, self.p, self.scale = model, features, p, scale
        self.network_evaluations = 0

    def forward(self, noisy, timesteps, features):
        if features is not self.features:
            raise ValueError('p-CFG sampling context differs from its bound H/static features')
        present = torch.ones(noisy.shape[0], dtype=torch.bool, device=noisy.device)
        if self.scale == 1.:
            self.network_evaluations += 1
            return self.model(noisy, timesteps, features, self.p, present)
        if self.scale == 0.:
            self.network_evaluations += 1
            return self.model(noisy, timesteps, features, self.p, ~present)
        self.network_evaluations += 2
        null = self.model(noisy, timesteps, features, self.p, ~present)
        conditional = self.model(noisy, timesteps, features, self.p, present)
        return null + self.scale * (conditional - null)


def cfg_sample(model, schedule, features, p, initial_noise, *, scale=1., steps=50, return_trace=False):
    """Same-network classifier-free v prediction, followed by eta=0 DDIM.

    scale=1 calls only the conditional branch and is bitwise equal to raw
    conditional sampling; scale=0 calls only null. Other finite nonnegative
    scales call both branches (two network evaluations per DDIM step). There
    is no extra random noise, physical-risk callback, clipping, or candidate
    selection. return_trace=True adds measured network-evaluation counts to
    the frozen sampler's trace dict; otherwise returns the coefficient tensor.
    """
    if (not isinstance(model, ClassifierFreePercentileDenoiser) or isinstance(scale, bool)
            or not math.isfinite(float(scale)) or float(scale) < 0):
        raise ValueError('p-CFG model and a finite nonnegative CFG scale required')
    adapter = _CFGPrediction(model, features, p, float(scale))
    result = ddim_sample(adapter, schedule, features, initial_noise, steps=steps,
                         prediction_type='v', return_trace=return_trace)
    if return_trace:
        result.update(CFG_scale=float(scale), network_evaluations=adapter.network_evaluations,
                      network_evaluations_per_step=1 if float(scale) in (0., 1.) else 2,
                      extra_sampling_noise=False, risk_oracle_calls=0,
                      output_is_final_single_path_not_selected_candidate=True)
    return result
