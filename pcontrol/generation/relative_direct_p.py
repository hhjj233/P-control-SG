"""History-only relative-message residuals for the existing p-CFG denoiser.

The inherited FiLM blocks, parameter names, null-p vector and arithmetic are
preserved. After each block, a separate directed actor-message branch reads
relative *observed* history and dynamic noisy-state tokens. Its output layer is
zero initialized; no conditional skip or detached gate hides it from autograd.
Thus a strict CFG-parent warm start initially reproduces the parent function,
while the new output layers can learn on the very first update.

Geometry uses the caller's existing FIT-normalized road-axis coordinates, not
metres or rotated agent frames. There is no new normalization, future geometry,
PET/CDF input, actor identifier, nearest-neighbor truncation or inference oracle.
This architecture is a candidate, not a claim of improved control or safety.
"""
import math
from collections.abc import Mapping

import torch
from torch import nn

from .direct_p_cfg import ClassifierFreePercentileDenoiser
from .diffusion import _clean_coefficients, _timesteps


EDGE_FEATURE_DIM = 18


def _observed_edges(history, dimensions, ego, mask):
    """Directed receiver i / sender j features, in external normalized units.

    Channels: (state_j-state_i) at t0 [4], its H13 mean [4], its
    (t0-minus-first-observation) change [4], dimensions_i [2], dimensions_j
    [2], ego_i [1], ego_j [1]. All history states were sanitized by the parent
    feature validator BEFORE pairwise arithmetic. Self edges and padding have
    zero features and zero weight. H13 change is not mislabeled acceleration.
    """
    current = history[:, -1]
    mean = history.mean(1)
    change = current - history[:, 0]
    count = mask.shape[1]
    # (B, receiver i, sender j, channel); sign is always sender minus receiver.
    differences = [value[:, None] - value[:, :, None]
                   for value in (current, mean, change)]
    receiver_dimensions = dimensions[:, :, None].expand(-1, -1, count, -1)
    sender_dimensions = dimensions[:, None].expand(-1, count, -1, -1)
    role = ego.to(history.dtype)
    receiver_role = role[:, :, None, None].expand(-1, -1, count, -1)
    sender_role = role[:, None, :, None].expand(-1, count, -1, -1)
    edges = torch.cat((*differences, receiver_dimensions, sender_dimensions,
                       receiver_role, sender_role), -1)
    pair_mask = (mask[:, :, None] & mask[:, None, :]
                 & ~torch.eye(count, dtype=torch.bool, device=mask.device)[None])
    edges = torch.where(pair_mask[..., None], edges, torch.zeros_like(edges))
    if not bool(torch.isfinite(edges).all()):
        raise ValueError('observed relative edge arithmetic must remain finite')
    return edges, pair_mask


class _RelativeMessageResidual(nn.Module):
    """Multihead dynamic attention plus observed-geometry messages.

    Geometry creates both logit bias and messages. Edge latents are shared
    across heads, avoiding a [B,N,N,hidden_dim] key/value tensor. A fully masked
    receiver has exactly zero weights and residual, including after training.
    Standard matmul/softmax operations retain Torch 1.12 double backward.
    """
    def __init__(self, width, heads, edge_hidden_dim):
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.token_norm = nn.LayerNorm(width)
        self.modulation = nn.Linear(2 * width, 2 * width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.edge_encoder = nn.Sequential(nn.Linear(EDGE_FEATURE_DIM, edge_hidden_dim),
                                          nn.SiLU(), nn.Linear(edge_hidden_dim, edge_hidden_dim))
        self.edge_bias = nn.Linear(edge_hidden_dim, heads)
        self.edge_value = nn.Linear(heads * edge_hidden_dim, width)
        self.output = nn.Linear(width, width)
        self.zero_output()

    def zero_output(self):
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, tokens, mask, edges, pair_mask, p_embedding, time_embedding):
        batch, count, width = tokens.shape
        scale, shift = self.modulation(torch.cat((p_embedding, time_embedding), -1)).chunk(2, -1)
        clean = torch.where(mask[..., None], tokens, torch.zeros_like(tokens))
        modulated = self.token_norm(clean) * (1. + scale[:, None]) + shift[:, None]

        def split(value):
            return value.reshape(batch, count, self.heads, self.head_width).transpose(1, 2)

        query, key, value = (split(layer(modulated)) for layer in (self.query, self.key, self.value))
        encoded_edges = self.edge_encoder(edges)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_width)
        scores = scores + self.edge_bias(encoded_edges).permute(0, 3, 1, 2)
        valid = pair_mask[:, None]
        # A finite sentinel, then explicit masking, avoids NaN softmax rows for
        # padded receivers and the valid singleton case (no other actor).
        weights = torch.softmax(scores.masked_fill(~valid, -torch.finfo(scores.dtype).max), -1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        dynamic_message = torch.matmul(weights, value).transpose(1, 2).reshape(batch, count, width)
        edge_message = torch.einsum('bhij,bije->bhie', weights, encoded_edges)
        edge_message = edge_message.permute(0, 2, 1, 3).reshape(batch, count, -1)
        message = dynamic_message + self.edge_value(edge_message)
        residual = self.output(torch.nn.functional.silu(message))
        recipients = mask & pair_mask.any(-1)
        residual = torch.where(recipients[..., None], residual, torch.zeros_like(residual))
        return torch.where(mask[..., None], clean + residual, torch.zeros_like(clean))


class RelativeInteractionPercentileDenoiser(ClassifierFreePercentileDenoiser):
    """p-CFG-compatible, variable-N denoiser with observed relative messages.

    Use load_from_parent_state_dict for a trained CFG/tangent parent. Loading
    an already-relative checkpoint uses ordinary strict load_state_dict. This
    module has no file access: an external caller must verify checkpoint hash
    and training/selection lineage before supplying the parent's state dict.
    """
    VERSION = 'natural_direct_p_CFG_observed_relative_message_residual_v1'

    def __init__(self, coefficient_dim, hidden_dim=128, heads=4, layers=3,
                 feedforward_dim=256, edge_hidden_dim=32):
        if type(edge_hidden_dim) is not int or edge_hidden_dim < 1:
            raise ValueError('edge_hidden_dim must be a positive integer')
        super().__init__(coefficient_dim, hidden_dim=hidden_dim, heads=heads,
                         layers=layers, feedforward_dim=feedforward_dim)
        self.edge_hidden_dim = edge_hidden_dim
        self.relative_blocks = nn.ModuleList([
            _RelativeMessageResidual(hidden_dim, heads, edge_hidden_dim) for _ in range(layers)])

    def load_from_direct_p_state_dict(self, state_dict):
        raise ValueError('relative model requires a trained CFG parent including learned null_p_embedding')

    def load_from_parent_state_dict(self, state_dict):
        """Validate every parent key/shape/dtype/finite value before mutation.

        All parent weights, including its learned null, are copied unchanged.
        The only permitted extra model keys are relative_blocks.*. New output
        layers are reset to zero on this explicit warm-start operation; no
        other new branch weights are reinitialized and no RNG is consumed.
        """
        expected = self.state_dict()
        extra = {name for name in expected if name.startswith('relative_blocks.')}
        parent = set(expected) - extra
        if not isinstance(state_dict, Mapping) or set(state_dict) != parent:
            raise ValueError('CFG parent must supply every and only inherited tensor, including learned null')
        for name in parent:
            tensor = state_dict[name]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape
                    or tensor.dtype != expected[name].dtype or not bool(torch.isfinite(tensor).all())):
                raise ValueError('CFG parent tensor shape/dtype/finite contract failed: ' + name)
        incompatible = super().load_state_dict(state_dict, strict=False)
        if set(incompatible.missing_keys) != extra or incompatible.unexpected_keys:
            raise RuntimeError('only the declared relative branch keys may be absent from a parent')
        for block in self.relative_blocks:
            block.zero_output()
        return dict(parent_state_loaded=True, learned_parent_null_retained=True,
                    new_tensor_keys=sorted(extra), new_output_layers_zeroed=True,
                    original_FiLM_blocks_unchanged=True,
                    checkpoint_hash_verification='caller_responsibility_not_performed_here')

    def architecture_config(self):
        config = super().architecture_config()
        config.update(version=self.VERSION, edge_hidden_dim=self.edge_hidden_dim,
            relative_message_residual=True, relative_branch_count=len(self.relative_blocks),
            relative_branch_parameter_count=sum(p.numel() for p in self.relative_blocks.parameters()),
            edge_feature_dim=EDGE_FEATURE_DIM,
            edge_features='sender_minus_receiver_t0_H13mean_H13change_xyvxvy_dims_i_dims_j_ego_i_ego_j',
            edge_coordinates='external_FIT_normalized_road_axes_not_physical_units',
            edge_source='observed_H13_static_only_no_GT_future_or_future_extrapolation',
            edge_self_loops=False, edge_all_valid_actor_pairs=True, edge_neighbor_truncation=False,
            geometry_message_and_attention_bias=True, dynamic_noisy_token_messages=True,
            relative_branch_p_time_FiLM=True, relative_branch_output_zero_initialized=True,
            zero_initialization_does_not_skip_or_detach_branch=True,
            learned_parent_null_retained_on_CFG_warm_start=True,
            null_initialization='constructor_midpoint_only; CFG_warm_start_preserves_learned_parent_null',
            null_injection='input_original_FiLM_and_relative_message_FiLM',
            weight_provenance='external_CFG_parent_binding_or_relative_checkpoint_reload',
            empirical_performance_improvement_verified=False)
        return config

    def relative_edge_features(self, features):
        """Inspectable history-only edge construction; no noisy/p/risk input."""
        history, dimensions, _road, mask, ego, _road_mask = self._clean_features(features)
        return _observed_edges(history, dimensions, ego, mask)

    def forward(self, noisy_coefficients, timesteps, features, p, condition_present=None):
        # Keep parent operation ordering intact; do not alter its MHA masks or
        # install hooks. The only additions are geometry and residual messages.
        h, dims, road, mask, ego, road_mask = self._clean_features(features)
        noisy = _clean_coefficients(noisy_coefficients, mask, name='noisy coefficients')
        if (noisy.dtype != torch.float32 or noisy.device != h.device
                or noisy.shape[:2] != mask.shape or math.prod(noisy.shape[2:]) != self.coefficient_dim):
            raise ValueError('float32 noisy coefficients must match the relative p-CFG denoiser')
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
        edges, pair_mask = _observed_edges(h, dims, ego, mask)
        for block, relative in zip(self.blocks, self.relative_blocks):
            tokens = block(tokens, mask, percentile, time)
            tokens = relative(tokens, mask, edges, pair_mask, percentile, time)
        prediction = self.output(tokens)
        prediction = torch.where(mask[..., None], prediction, torch.zeros_like(prediction))
        return prediction.reshape_as(noisy_coefficients)
