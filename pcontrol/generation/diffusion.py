"""Small independent history-conditioned joint-agent DDPM / DDIM backend.

This is an original implementation of standard epsilon-prediction DDPM
(Ho et al., 2020), optional velocity prediction (Salimans & Ho, 2022), the
cosine schedule (Nichol & Dhariwal, 2021), and eta=0 DDIM updates
(Song et al., 2021); no legacy MADiff/N9 code or weights are
copied. References: arxiv.org/abs/2006.11239, arxiv.org/abs/2102.09672,
arxiv.org/abs/2010.02502, arxiv.org/abs/2202.00512. Multi-agent attention has
no slot/identity embedding. Velocity prediction is only a parameterization;
this backend does not perform progressive distillation.

Only observed history/static context enters the denoiser. Coefficient targets
and their normalization/basis are external. This module does not import or
fit a risk estimator, accept percentile labels, or select among samples.
"""
import math
from typing import Mapping

import torch
from torch import nn


FEATURE_KEYS = frozenset(('history', 'dimensions', 'road_boundaries',
                          'road_boundary_mask', 'ego_mask', 'agent_mask'))


def _prediction_type(value):
    if value not in ('epsilon', 'v'):
        raise ValueError('prediction_type must be epsilon or v')
    return value


def _clean_coefficients(values, agent_mask, *, name='coefficients'):
    if (not isinstance(values, torch.Tensor) or values.ndim not in (3, 4)
            or not values.is_floating_point() or values.shape[0] < 1
            or values.shape[1] < 1 or any(size < 1 for size in values.shape[2:])
            or (values.ndim == 4 and values.shape[-1] != 2)):
        raise ValueError(name + ' require floating [B,N,D] or [B,N,K,2]')
    if (not isinstance(agent_mask, torch.Tensor) or agent_mask.dtype != torch.bool
            or agent_mask.shape != values.shape[:2] or agent_mask.device != values.device
            or not bool(agent_mask.any(1).all())):
        raise ValueError('boolean nonempty agent mask must match every scene')
    expanded = agent_mask.reshape(*agent_mask.shape, *([1] * (values.ndim - 2)))
    clean = torch.where(expanded, values, torch.zeros_like(values))
    if not bool(torch.isfinite(clean).all()):
        raise ValueError('valid ' + name + ' must be finite')
    return clean


def _timesteps(timesteps, batch, device, steps=None):
    if (not isinstance(timesteps, torch.Tensor) or timesteps.shape != (batch,)
            or timesteps.dtype != torch.long or timesteps.device != device
            or bool((timesteps < 0).any())
            or (steps is not None and bool((timesteps >= steps).any()))):
        raise ValueError('timesteps must be device-matched int64[B] within schedule')
    return timesteps


class _AttentionBlock(nn.Module):
    """Standard pre-norm self-attention and pointwise feed-forward residuals."""
    def __init__(self, width, heads, feedforward_dim):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, feedforward_dim), nn.GELU(),
                                         nn.Linear(feedforward_dim, width))

    def forward(self, tokens, mask):
        clean = torch.where(mask[..., None], tokens, torch.zeros_like(tokens))
        query = self.attention_norm(clean)
        attended, _ = self.attention(query, query, query, key_padding_mask=~mask,
                                     need_weights=False)
        clean = torch.where(mask[..., None], clean + attended, torch.zeros_like(clean))
        clean = clean + self.feedforward(self.feedforward_norm(clean))
        return torch.where(mask[..., None], clean, torch.zeros_like(clean))


class JointHistoryDenoiser(nn.Module):
    """Unordered multi-agent predictor with a single marked ego.

    Six feature keys match the scene reference interface exactly: externally
    normalized float32 history[B,13,N,4], dimensions[B,N,2], road_boundaries[B,R],
    and boolean road_boundary_mask[B,R], ego_mask[B,N], agent_mask[B,N]. There
    is exactly one valid ego, at ANY index. No coefficient/feature scaling is
    performed here. Background permutation and padding do not change outputs
    on corresponding actual vehicles (up to floating-point reduction error).
    """
    VERSION = 'independent_joint_history_DDPM_predictor_v1'

    def __init__(self, coefficient_dim, hidden_dim=128, heads=4, layers=3,
                 feedforward_dim=256):
        super().__init__()
        if (type(coefficient_dim) is not int or coefficient_dim < 2 or coefficient_dim % 2
                or type(hidden_dim) is not int or hidden_dim < 4 or hidden_dim % 2
                or type(heads) is not int or heads < 1 or hidden_dim % heads
                or type(layers) is not int or layers < 1
                or type(feedforward_dim) is not int or feedforward_dim < 1):
            raise ValueError('positive even coefficient/hidden widths and valid attention dimensions required')
        self.coefficient_dim = coefficient_dim
        self.hidden_dim, self.heads = hidden_dim, heads
        self.layers, self.feedforward_dim = layers, feedforward_dim
        self.history_encoder = nn.Sequential(nn.Linear(55, hidden_dim), nn.GELU(),
                                             nn.Linear(hidden_dim, hidden_dim))
        self.coefficient_encoder = nn.Linear(coefficient_dim, hidden_dim)
        self.time_encoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                          nn.Linear(hidden_dim, hidden_dim))
        self.road_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(),
                                          nn.Linear(hidden_dim, hidden_dim))
        self.road_pool_encoder = nn.Linear(2 * hidden_dim, hidden_dim)
        self.count_encoder = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList([_AttentionBlock(hidden_dim, heads, feedforward_dim)
                                     for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, coefficient_dim))
        frequencies = torch.exp(-math.log(10000.) * torch.arange(hidden_dim // 2,
                                     dtype=torch.float32) / (hidden_dim // 2 - 1))
        self.register_buffer('time_frequencies', frequencies)

    def architecture_config(self):
        return dict(version=self.VERSION, coefficient_dim=self.coefficient_dim,
            hidden_dim=self.hidden_dim, heads=self.heads, layers=self.layers,
            feedforward_dim=self.feedforward_dim, dropout=0.,
            parameter_count=sum(p.numel() for p in self.parameters()),
            feature_keys=sorted(FEATURE_KEYS), history_frames=13,
            unordered_background=True, ego_at_any_valid_index=True,
            slot_embeddings=False, actor_ID_input=False, focal_input=False,
            actual_future_input=False, risk_or_percentile_input=False,
            normalization='external_training_only; no_internal_second_scaling',
            supported_prediction_types=['epsilon', 'v'],
            prediction_type_binding='external_training_policy', pretrained_weights_loaded=False)

    def _clean_features(self, features):
        if not isinstance(features, Mapping) or set(features) != FEATURE_KEYS:
            raise ValueError('exactly the six declared observed-history/static feature keys required')
        if any(not isinstance(v, torch.Tensor) for v in features.values()):
            raise TypeError('all feature values must be tensors')
        h, dims, road = (features[k] for k in ('history', 'dimensions', 'road_boundaries'))
        mask, ego, road_mask = (features[k] for k in ('agent_mask', 'ego_mask', 'road_boundary_mask'))
        if (h.ndim != 4 or h.shape[0] < 1 or h.shape[1] != 13
                or h.shape[2] < 1 or h.shape[3] != 4):
            raise ValueError('history shape must be [B,13,N,4]')
        batch, _, agents, _ = h.shape
        if (dims.shape != (batch, agents, 2) or mask.shape != (batch, agents)
                or ego.shape != mask.shape or road.ndim != 2 or road.shape[0] != batch
                or road.shape[1] < 2 or road_mask.shape != road.shape):
            raise ValueError('history/static feature shapes disagree')
        if any(v.device != self.time_frequencies.device for v in features.values()):
            raise ValueError('features must share the model device')
        if any(v.dtype != torch.float32 for v in (h, dims, road)):
            raise ValueError('history/static inputs require externally normalized float32')
        if any(v.dtype != torch.bool for v in (mask, ego, road_mask)):
            raise ValueError('padding/ego/road masks must be boolean')
        if (bool((ego & ~mask).any()) or not bool((ego.sum(1) == 1).all())
                or not bool((road_mask.sum(1) >= 2).all())):
            raise ValueError('one actual ego and at least two actual road boundaries required')
        h = torch.where(mask[:, None, :, None], h, torch.zeros_like(h))
        dims = torch.where(mask[..., None], dims, torch.zeros_like(dims))
        road = torch.where(road_mask, road, torch.zeros_like(road))
        if (not all(bool(torch.isfinite(v).all()) for v in (h, dims, road))
                or bool((dims[mask] <= 0).any())):
            raise ValueError('actual observations must be finite with positive dimensions')
        return h, dims, road, mask, ego, road_mask

    def forward(self, noisy_coefficients, timesteps, features):
        h, dims, road, mask, ego, road_mask = self._clean_features(features)
        noisy = _clean_coefficients(noisy_coefficients, mask, name='noisy coefficients')
        if (noisy.dtype != torch.float32 or noisy.device != h.device
                or noisy.shape[:2] != mask.shape
                or math.prod(noisy.shape[2:]) != self.coefficient_dim):
            raise ValueError('float32 noisy coefficient dimensions must match the denoiser')
        batch, count = mask.shape
        t = _timesteps(timesteps, batch, noisy.device)
        history = h.permute(0, 2, 1, 3).reshape(batch, count, 52)
        observed = self.history_encoder(torch.cat((history, dims, ego[..., None].to(h.dtype)), -1))
        road_tokens = self.road_encoder(road[..., None])
        road_mean = torch.where(road_mask[..., None], road_tokens, torch.zeros_like(road_tokens)).sum(1)
        road_mean = road_mean / road_mask.sum(1).to(h.dtype)[:, None]
        road_max = road_tokens.masked_fill(~road_mask[..., None], -torch.inf).max(1).values
        context = self.road_pool_encoder(torch.cat((road_mean, road_max), -1))
        context = context + self.count_encoder(torch.log1p(mask.sum(1).to(h.dtype))[:, None])
        phase = t.to(h.dtype)[:, None] * self.time_frequencies[None]
        time = self.time_encoder(torch.cat((torch.sin(phase), torch.cos(phase)), -1))
        tokens = observed + self.coefficient_encoder(noisy.reshape(batch, count, self.coefficient_dim))
        tokens = tokens + context[:, None] + time[:, None]
        for block in self.blocks:
            tokens = block(tokens, mask)
        epsilon = self.output(tokens)
        epsilon = torch.where(mask[..., None], epsilon, torch.zeros_like(epsilon))
        return epsilon.reshape_as(noisy_coefficients)


class CosineDiffusionSchedule(nn.Module):
    """Nichol cosine schedule; t=0 is first noisy step, clean endpoint is t=-1.

    alpha_bar(t/T)=cos²((t/T+s)/(1+s)*pi/2)/cos²(s/(1+s)*pi/2).
    beta_i=min(1-alpha_bar((i+1)/T)/alpha_bar(i/T), .999).
    Buffers are float64; coefficients are cast to each state tensor's dtype.
    """
    def __init__(self, steps=100, offset=.008, max_beta=.999):
        super().__init__()
        if (type(steps) is not int or steps < 2 or not math.isfinite(offset) or offset < 0
                or not math.isfinite(max_beta) or not 0 < max_beta < 1):
            raise ValueError('valid cosine schedule parameters required')
        self.steps, self.offset, self.max_beta = steps, float(offset), float(max_beta)
        grid = torch.arange(steps + 1, dtype=torch.float64) / steps
        curve = torch.cos((grid + offset) / (1 + offset) * math.pi / 2).square()
        curve = curve / curve[0]
        betas = (1. - curve[1:] / curve[:-1]).clamp(max=max_beta)
        if not bool(((betas > 0) & (betas < 1)).all()):
            raise ValueError('cosine parameters produced a degenerate schedule')
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(1. - betas, 0))

    def alpha_bar(self, timesteps, like):
        _timesteps(timesteps, like.shape[0], like.device, self.steps)
        if self.alphas_cumprod.device != like.device:
            raise ValueError('schedule and state must share a device; move schedule with .to(device)')
        return self.alphas_cumprod[timesteps].to(like.dtype).reshape(
            like.shape[0], *([1] * (like.ndim - 1)))

    def q_sample(self, x0, timesteps, noise, agent_mask):
        clean = _clean_coefficients(x0, agent_mask)
        noise = _clean_coefficients(noise, agent_mask, name='noise')
        if noise.shape != clean.shape or noise.dtype != clean.dtype:
            raise ValueError('noise and clean state must have identical shape/dtype')
        alpha = self.alpha_bar(timesteps, clean)
        return alpha.sqrt() * clean + (1. - alpha).sqrt() * noise

    def predict_x0(self, x_t, timesteps, epsilon, agent_mask):
        x_t = _clean_coefficients(x_t, agent_mask)
        epsilon = _clean_coefficients(epsilon, agent_mask, name='epsilon')
        if epsilon.shape != x_t.shape or epsilon.dtype != x_t.dtype:
            raise ValueError('epsilon and noisy state must have identical shape/dtype')
        alpha = self.alpha_bar(timesteps, x_t)
        return (x_t - (1. - alpha).sqrt() * epsilon) / alpha.sqrt()

    def velocity_target(self, x0, timesteps, noise, agent_mask):
        """v=√A*epsilon−√(1−A)*x0, without changing the forward noise law."""
        clean = _clean_coefficients(x0, agent_mask)
        noise = _clean_coefficients(noise, agent_mask, name='noise')
        if clean.shape != noise.shape or clean.dtype != noise.dtype:
            raise ValueError('clean state and noise must have identical shape/dtype')
        alpha = self.alpha_bar(timesteps, clean)
        return alpha.sqrt() * noise - (1. - alpha).sqrt() * clean

    def prediction_to_x0_epsilon(self, x_t, timesteps, prediction, agent_mask,
                                prediction_type='epsilon'):
        """Resolve an explicit parameterization; velocity inversion has no /√A.

        For v: x0=√A*x_t−√(1−A)*v; epsilon=√A*v+√(1−A)*x_t.
        This is the inverse orthogonal rotation of forward x_t and target v.
        """
        _prediction_type(prediction_type)
        x_t = _clean_coefficients(x_t, agent_mask)
        prediction = _clean_coefficients(prediction, agent_mask, name='model prediction')
        if prediction.shape != x_t.shape or prediction.dtype != x_t.dtype:
            raise ValueError('prediction and noisy state must have identical shape/dtype')
        if prediction_type == 'epsilon':
            return self.predict_x0(x_t, timesteps, prediction, agent_mask), prediction
        alpha = self.alpha_bar(timesteps, x_t)
        x0 = alpha.sqrt() * x_t - (1. - alpha).sqrt() * prediction
        epsilon = alpha.sqrt() * prediction + (1. - alpha).sqrt() * x_t
        return x0, epsilon

    def ddim_timesteps(self, steps=50):
        if type(steps) is not int or not 1 <= steps <= self.steps:
            raise ValueError('DDIM step count must be between 1 and the DDPM horizon')
        return torch.linspace(self.steps - 1, 0, steps, device=self.betas.device).round().long()

    def as_dict(self):
        return dict(schedule='nichol_cosine', steps=self.steps, offset=self.offset,
                    max_beta=self.max_beta, timestep_zero_is_first_noisy=True,
                    beta_values=self.betas.detach().cpu().tolist())


def scene_mean_epsilon_mse(prediction, noise, agent_mask, reduction='mean'):
    """Mean per actual coefficient, then mean per scene; padding has zero weight."""
    prediction = _clean_coefficients(prediction, agent_mask, name='prediction')
    noise = _clean_coefficients(noise, agent_mask, name='noise')
    if prediction.shape != noise.shape or prediction.dtype != noise.dtype:
        raise ValueError('epsilon prediction and noise must have equal shape/dtype')
    coefficient_count = math.prod(prediction.shape[2:])
    per_scene = (prediction - noise).square().reshape(prediction.shape[0], -1).sum(1)
    per_scene = per_scene / (agent_mask.sum(1).to(prediction.dtype) * coefficient_count)
    if reduction == 'none':
        return per_scene
    if reduction == 'mean':
        return per_scene.mean()
    raise ValueError('reduction must be mean or none')


def diffusion_training_loss(model, schedule, x0, features, *, timesteps=None,
                            noise=None, generator=None, prediction_type='epsilon'):
    """Uniform DDPM time; equal-scene MSE on explicit epsilon or velocity target.

    v-MSE is not asserted to be the same time weighting as epsilon-MSE. The
    experiment must bind the chosen parameterization before training.
    """
    _prediction_type(prediction_type)
    mask = features['agent_mask']
    clean = _clean_coefficients(x0, mask)
    if timesteps is None:
        timesteps = torch.randint(schedule.steps, (clean.shape[0],), device=clean.device,
                                  generator=generator)
    if noise is None:
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    noise = _clean_coefficients(noise, mask, name='noise')
    x_t = schedule.q_sample(clean, timesteps, noise, mask)
    prediction = model(x_t, timesteps, features)
    target = noise if prediction_type == 'epsilon' else schedule.velocity_target(clean, timesteps, noise, mask)
    per_scene = scene_mean_epsilon_mse(prediction, target, mask, reduction='none')
    epsilon = prediction if prediction_type == 'epsilon' else schedule.prediction_to_x0_epsilon(
        x_t, timesteps, prediction, mask, prediction_type='v')[1]
    return dict(loss=per_scene.mean(), per_scene_loss=per_scene,
                timesteps=timesteps, noise=noise, noisy_coefficients=x_t,
                model_prediction=prediction, prediction_target=target,
                prediction_type=prediction_type, epsilon_prediction=epsilon)


def ddim_sample(model, schedule, features, initial_noise, *, steps=50,
                x0_callback=None, return_trace=False, prediction_type='epsilon'):
    """Deterministic eta=0 DDIM, exactly one path from caller-supplied z.

    Epsilon mode uses x0=(x_t-sqrt(1-A_t)*eps)/sqrt(A_t). Velocity mode uses
    x0=sqrt(A_t)*x_t-sqrt(1-A_t)*v and
    eps=sqrt(A_t)*v+sqrt(1-A_t)*x_t, without division by sqrt(A_t). Then
    x_s=sqrt(A_s)*x0+sqrt(1-A_s)*eps; final s=-1 has A_s=1.
    No clipping, resampling, best-of-K selection or implicit RNG is performed.

    Optional callback(x0_prediction, timestep[B], x_t) returns a same-shaped
    processed x0. It runs outside the denoiser's no_grad block so the external
    caller can construct its own local gradient; no denoiser graph is kept.
    After processing, epsilon is rederived to remain consistent with x_t;
    an exactly unchanged x0 retains the original epsilon to make a zero-effect
    callback bitwise identical to the uncontrolled path.
    The callback is the sole generic extension point: this module has no risk
    dependency. Call model.eval() externally; model mode is not mutated here.
    """
    _prediction_type(prediction_type)
    mask = features['agent_mask']
    x_t = _clean_coefficients(initial_noise, mask, name='initial noise').detach().clone()
    indices = schedule.ddim_timesteps(steps)
    trace = []
    for position, index in enumerate(indices.tolist()):
        timesteps = torch.full((x_t.shape[0],), index, device=x_t.device, dtype=torch.long)
        with torch.no_grad():
            prediction = model(x_t, timesteps, features)
            x0, epsilon = schedule.prediction_to_x0_epsilon(
                x_t, timesteps, prediction, mask, prediction_type=prediction_type)
        if x0_callback is not None:
            processed = x0_callback(x0.detach().clone(), timesteps.clone(), x_t.detach().clone())
            if (not isinstance(processed, torch.Tensor) or processed.shape != x0.shape
                    or processed.dtype != x0.dtype or processed.device != x0.device):
                raise ValueError('x0 callback must return same-shaped/dtype/device coefficients')
            processed = _clean_coefficients(processed, mask, name='callback x0').detach()
            if not torch.equal(processed, x0):
                x0 = processed
                alpha = schedule.alpha_bar(timesteps, x_t)
                epsilon = (x_t - alpha.sqrt() * x0) / (1. - alpha).sqrt()
        next_index = int(indices[position + 1]) if position + 1 < len(indices) else -1
        with torch.no_grad():
            if next_index == -1:
                x_next = x0
            else:
                previous = torch.full_like(timesteps, next_index)
                alpha_previous = schedule.alpha_bar(previous, x_t)
                x_next = alpha_previous.sqrt() * x0 + (1. - alpha_previous).sqrt() * epsilon
            x_t = _clean_coefficients(x_next, mask, name='DDIM state').detach()
        if return_trace:
            trace.append(dict(timestep=index, next_timestep=next_index,
                              x0_prediction=x0.detach().clone(), state=x_t.clone()))
    return dict(sample=x_t, timesteps=indices.detach().clone(), trace=trace,
                prediction_type=prediction_type) if return_trace else x_t
