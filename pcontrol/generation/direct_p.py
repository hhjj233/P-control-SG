"""Direct-P: requested continuous percentile is an actual denoiser condition.

This new backend reuses the frozen history/mask utilities, cosine schedule and
standard diffusion algebra, but never loads the earlier generator's weights.
Each transformer block modulates both pre-normalized attention and feedforward
inputs with p and diffusion-time FiLM. The same p embedding also enters the
initial actor tokens. Actor order remains an unordered set except for ego role.

Inference receives H/static context and requested p, NOT a target PET, a risk
estimator, or an external correction. Public loss/sample helpers use velocity
prediction; training-label construction and trajectory basis remain external.
"""
import math

import torch
from torch import nn

from .diffusion import (JointHistoryDenoiser, _clean_coefficients, _timesteps,
                        diffusion_training_loss, ddim_sample)


def _percentile(p, batch, device):
    if (batch < 1 or not isinstance(p, torch.Tensor) or p.shape != (batch,)
            or p.dtype not in (torch.float32, torch.float64) or p.device != device
            or not bool(torch.isfinite(p).all()) or bool(((p < 0) | (p > 1)).any())):
        raise ValueError('p must be a finite device-matched float[B] in [0,1]')
    return p


class _PercentileTimeFiLMBlock(nn.Module):
    """Pre-norm self-attention/FFN, with scene-shared p/time affine modulation."""
    def __init__(self, width, heads, feedforward_dim):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, feedforward_dim), nn.GELU(),
                                         nn.Linear(feedforward_dim, width))
        self.modulation = nn.Linear(2 * width, 4 * width)

    def forward(self, tokens, mask, p_embedding, time_embedding):
        scale_a, shift_a, scale_f, shift_f = self.modulation(
            torch.cat((p_embedding, time_embedding), -1)).chunk(4, -1)
        clean = torch.where(mask[..., None], tokens, torch.zeros_like(tokens))
        query = self.attention_norm(clean) * (1. + scale_a[:, None]) + shift_a[:, None]
        attended, _ = self.attention(query, query, query, key_padding_mask=~mask, need_weights=False)
        clean = torch.where(mask[..., None], clean + attended, torch.zeros_like(clean))
        feed = self.feedforward_norm(clean) * (1. + scale_f[:, None]) + shift_f[:, None]
        clean = clean + self.feedforward(feed)
        return torch.where(mask[..., None], clean, torch.zeros_like(clean))


class JointPercentileDenoiser(JointHistoryDenoiser):
    """Fresh direct p-conditioned, joint unordered-agent velocity predictor.

    forward(noisy[B,N,K,2] or [B,N,2K], t[B], features, p[B]) has the exact six
    history/static keys accepted by the independent prior, plus a SEPARATE p
    tensor. No actor IDs, focal role, future risk/semantics or target PET field
    is accepted. p is not rounded, bucketed, detached, or converted to a target
    PET inside the model. Float64 p is cast to float32 with its gradient intact.
    All coefficient/history/static normalization remains external and FIT-only.
    """
    VERSION = 'natural_direct_continuous_p_joint_FiLM_denoiser_v1'

    def __init__(self, coefficient_dim, hidden_dim=128, heads=4, layers=3,
                 feedforward_dim=256):
        # Reuse only definitions and fresh initialization, never old weights.
        super().__init__(coefficient_dim, hidden_dim=hidden_dim, heads=heads,
                         layers=layers, feedforward_dim=feedforward_dim)
        self.p_encoder = nn.Sequential(nn.Linear(9, hidden_dim), nn.SiLU(),
                                        nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([_PercentileTimeFiLMBlock(hidden_dim, heads, feedforward_dim)
                                     for _ in range(layers)])
        self.register_buffer('p_frequencies', math.pi * torch.tensor([1., 2., 4., 8.]))

    def architecture_config(self):
        config = super().architecture_config()
        config.update(version=self.VERSION, p_condition_is_actual_model_input=True,
            p_shape='one_continuous_scalar_per_scene_B', p_range=[0., 1.], p_rounding=False,
            p_embedding='centered_2p_minus1_plus_sin_cos_pi_times_1_2_4_8_then_shared_MLP',
            p_feature_dim=9, p_input_token_injection=True,
            p_time_FiLM='independent_scale_shift_after_each_block_attention_and_FFN_LayerNorm',
            FiLM_conditioning='concatenated_p_and_diffusion_time_embeddings',
            p_as_actor_ID_or_slot=False, target_PET_input=False,
            risk_estimator_dependency=False, external_guidance_in_primary_sampling=False,
            pretrained_weights_loaded=False, supported_prediction_types=['v'],
            prediction_type_binding='v_only_public_helpers',
            normalization='external_FIT_only; no_internal_second_scaling')
        # The base flag referred to its absent scalar-condition input. Override
        # it honestly for this new architecture rather than keeping stale metadata.
        config['risk_or_percentile_input'] = True
        return config

    def percentile_embedding(self, p):
        """Continuous differentiable embedding; centered scalar distinguishes endpoints."""
        _percentile(p, p.shape[0] if isinstance(p, torch.Tensor) and p.ndim else -1,
                    self.p_frequencies.device)
        centered = 2. * p.to(torch.float32) - 1.
        phase = centered[:, None] * self.p_frequencies[None]
        return self.p_encoder(torch.cat((centered[:, None], torch.sin(phase), torch.cos(phase)), -1))

    def forward(self, noisy_coefficients, timesteps, features, p):
        h, dims, road, mask, ego, road_mask = self._clean_features(features)
        noisy = _clean_coefficients(noisy_coefficients, mask, name='noisy coefficients')
        if (noisy.dtype != torch.float32 or noisy.device != h.device
                or noisy.shape[:2] != mask.shape or math.prod(noisy.shape[2:]) != self.coefficient_dim):
            raise ValueError('float32 noisy coefficient shape must match the direct-P denoiser')
        batch, count = mask.shape
        t = _timesteps(timesteps, batch, noisy.device)
        _percentile(p, batch, noisy.device)
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
        percentile = self.percentile_embedding(p)
        tokens = observed + self.coefficient_encoder(noisy.reshape(batch, count, self.coefficient_dim))
        tokens = tokens + context[:, None] + time[:, None] + percentile[:, None]
        for block in self.blocks:
            tokens = block(tokens, mask, percentile, time)
        prediction = self.output(tokens)
        prediction = torch.where(mask[..., None], prediction, torch.zeros_like(prediction))
        return prediction.reshape_as(noisy_coefficients)

    def bind_condition(self, p, features):
        """Bind separate p/H to the frozen diffusion helper's three-argument API.

        Tensors remain connected to autograd and must already be on this model's
        device. The adapter does not copy/reload weights or change model mode.
        Pass the same features mapping to frozen helper calls, not a new context.
        """
        h, _dims, _road, _mask, _ego, _road_mask = self._clean_features(features)
        _percentile(p, h.shape[0], h.device)
        return _BoundPercentileCondition(self, p, features)


class _BoundPercentileCondition(nn.Module):
    def __init__(self, model, p, features):
        super().__init__()
        self.model, self.p, self.features = model, p, features

    def forward(self, noisy, timesteps, features=None):
        if features is not None and features is not self.features:
            raise ValueError('adapter context differs from its explicitly bound history/static mapping')
        return self.model(noisy, timesteps, self.features, self.p)


# Naming compatibility for the separate new trainer; both names are this same
# fresh architecture, not an alias to the frozen unconditional generator.
DirectPJointHistoryDenoiser = JointPercentileDenoiser


def direct_p_training_loss(model, schedule, x0, features, p, *, timesteps=None,
                           noise=None, generator=None, prediction_type='v'):
    """Equal-scene standard v-MSE, with p supplied to the denoiser itself."""
    if not isinstance(model, JointPercentileDenoiser) or prediction_type != 'v':
        raise ValueError('direct-P public training requires the new conditioned model and v prediction')
    adapter = model.bind_condition(p, features)
    result = diffusion_training_loss(adapter, schedule, x0, features, timesteps=timesteps,
        noise=noise, generator=generator, prediction_type='v')
    result['conditioning_p'] = p
    return result


def direct_p_sample(model, schedule, features, p, initial_noise, *, steps=50, return_trace=False):
    """One eta=0 v-DDIM path, directly p-conditioned; no risk callback/correction.

    Initial noise is caller-supplied and not modified. Reuse the same tensor for
    paired p queries. Coefficient decoding and exact t0 anchoring are external.
    This API intentionally exposes no guidance or candidate-selection argument.
    """
    if not isinstance(model, JointPercentileDenoiser):
        raise ValueError('direct-P sampling requires the actual p-conditioned denoiser')
    return ddim_sample(model.bind_condition(p, features), schedule, features, initial_noise,
                       steps=steps, prediction_type='v', return_trace=return_trace)
