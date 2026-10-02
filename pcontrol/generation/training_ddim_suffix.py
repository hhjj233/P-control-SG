"""Training-only last-K backpropagation through an otherwise unchanged DDIM path.

All sampling steps run from the caller's single noise tensor. The prefix is
detached, and the last K predictions AND state updates retain their graph.
This is a truncated model-parameter gradient when K < steps, not the full
sampler derivative. H, requested p and initial noise are fixed conditions.
No loss, optimizer, risk callback, candidate selection or data access is used.
The frozen inference sampler remains unchanged.
"""
import math

import torch
from torch.utils.checkpoint import checkpoint

from .direct_p_cfg import ClassifierFreePercentileDenoiser, _CFGPrediction
from .diffusion import CosineDiffusionSchedule, _clean_coefficients


def training_ddim_suffix(model, schedule, features, p, initial_noise, *, steps=50,
                         grad_last_steps=5, cfg_scale=4.):
    """Return final coefficients and explicit gradient/compute metadata.

    Accepts the same six H/static tensors, p[B] and float32 [B,N,K,2] or
    [B,N,D] noise as cfg_sample. All external condition tensors are detached
    without changing their values or requires_grad flags. Initial noise is
    additionally cloned; it is never optimized or mutated. The same network's
    conditional/null v predictions use the original CFG branch implementation.

    K=0 gives an entirely detached output; K=steps differentiates the full
    trajectory with respect to model parameters, holding H/p/z fixed. Other K
    values cut the state graph before the last K steps. An outer no_grad block
    is supported by explicitly enabling gradients in the suffix; inference_mode
    is refused because enable_grad cannot override its tensor semantics.

    Suffix network predictions use reentrant checkpointing: their no-grad
    forward matches the inference kernel path, and backward recomputes those
    predictions with gradients. Use loss.backward(), not autograd.grad with an
    explicit inputs argument; this first version supports first-order parameter
    training, not higher-order differentiation through the checkpoint. Backward
    recomputation adds K times the per-step CFG network-call count.

    Model train/eval mode, parameter values, RNG state and global gradient mode
    are not modified. The caller owns any loss/backward/optimizer operation.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError('training DDIM suffix cannot run inside torch.inference_mode')
    if not isinstance(model, ClassifierFreePercentileDenoiser):
        raise ValueError('a p-CFG-compatible denoiser is required')
    if not isinstance(schedule, CosineDiffusionSchedule):
        raise ValueError('the existing cosine diffusion schedule is required')
    if type(steps) is not int or not 1 <= steps <= schedule.steps:
        raise ValueError('steps must be an integer within the diffusion horizon')
    if type(grad_last_steps) is not int or not 0 <= grad_last_steps <= steps:
        raise ValueError('grad_last_steps must be an integer between zero and steps')
    if isinstance(cfg_scale, bool) or not math.isfinite(float(cfg_scale)) or float(cfg_scale) < 0:
        raise ValueError('cfg_scale must be finite and nonnegative')
    model._clean_features(features)
    if not isinstance(p, torch.Tensor):
        raise ValueError('p must be a tensor as in the original CFG sampler')
    context = {key: value.detach() for key, value in features.items()}
    condition = p.detach()
    mask = context['agent_mask']
    state = _clean_coefficients(initial_noise, mask, name='initial noise').detach().clone()
    indices = schedule.ddim_timesteps(steps)
    values = indices.tolist()
    prefix_steps = steps - grad_last_steps
    prediction = _CFGPrediction(model, context, condition, float(cfg_scale))
    def predict(current, current_times):
        return prediction(current, current_times, context)
    for position, index in enumerate(values):
        track = position >= prefix_steps
        if track and position == prefix_steps:
            # A private graph-boundary leaf makes reentrant checkpointing work
            # even though the public noise and prefix are deliberately detached.
            # This leaf is not the caller's noise and is never an optimizer input.
            state = state.detach().requires_grad_(True)
        # Both prediction and the update need to remain in the same graph.
        with torch.enable_grad() if track else torch.no_grad():
            timesteps = torch.full((state.shape[0],), index, dtype=torch.long, device=state.device)
            v = checkpoint(predict, state, timesteps, use_reentrant=True,
                           preserve_rng_state=True) if track else predict(state, timesteps)
            clean, epsilon = schedule.prediction_to_x0_epsilon(
                state, timesteps, v, mask, prediction_type='v')
            next_index = values[position + 1] if position + 1 < len(values) else -1
            if next_index == -1:
                next_state = clean
            else:
                previous = torch.full_like(timesteps, next_index)
                alpha = schedule.alpha_bar(previous, state)
                next_state = alpha.sqrt() * clean + (1. - alpha).sqrt() * epsilon
            state = _clean_coefficients(next_state, mask, name='DDIM state')
            if not track:
                state = state.detach()
    return dict(sample=state, timesteps=indices.detach().clone(), prediction_type='v',
        steps=steps, grad_last_steps=grad_last_steps, detached_prefix_steps=prefix_steps,
        gradient_timesteps=values[prefix_steps:],
        full_chain_gradient=grad_last_steps == steps,
        gradient_scope='model_parameters_only; H_p_initial_noise_are_fixed',
        truncated_parameter_gradient=0 < grad_last_steps < steps,
        sample_requires_grad=state.requires_grad,
        initial_noise_detached=True, history_static_detached=True, requested_p_detached=True,
        CFG_scale=float(cfg_scale), network_evaluations=prediction.network_evaluations,
        network_evaluations_per_step=1 if float(cfg_scale) in (0., 1.) else 2,
        backward_recompute_network_evaluations=grad_last_steps * (1 if float(cfg_scale) in (0., 1.) else 2),
        checkpointing='reentrant_no_grad_forward_grad_recompute_backward',
        backward_api='loss.backward; no explicit-input autograd.grad or higher-order contract',
        risk_oracle_calls=0, additional_sampling_noise=False, candidate_selection=False,
        optimizer_step_performed=False, model_mode_changed=False)
