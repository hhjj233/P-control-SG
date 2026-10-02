"""Fresh no-focal ego-scene CDF references, independent of frozen pair models.

All non-ego vehicles are an unordered set. Only the six declared history/static
feature keys enter a model; no semantics, actor IDs, future or PET is accepted.
M1 and M2 have identical trainable tensors and output scorers. They differ only
in whether relation-attention queries contain the fixed PET-range encoding.
The scalar outcome is the externally defined ego-vs-all scene PET, NOT a pair
label. This module neither computes that label nor loads pretrained weights.
"""
from typing import Mapping

import torch
from torch import nn

from .mixed_cdf import distribution_from_logits


FEATURE_KEYS = frozenset(('history', 'dimensions', 'road_boundaries',
                          'road_boundary_mask', 'ego_mask', 'agent_mask'))


def _mlp(input_dim, hidden, output):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                         nn.Linear(hidden, output), nn.LayerNorm(output))


def _masked_pool(values, mask):
    clean = torch.where(mask[..., None], values, torch.zeros_like(values))
    mean = clean.sum(1) / mask.sum(1).to(values.dtype)[:, None]
    maximum = clean.masked_fill(~mask[..., None], -torch.inf).max(1).values
    return mean, maximum


class SceneCDFReference(nn.Module):
    """M0 pooled; M1 static relation attention; M2 PET-range relation attention.

    Input tensors: history float32[B,13,N,4], dimensions float32[B,N,2],
    road_boundaries float32[B,R], and boolean agent_mask/ego_mask[B,N],
    road_boundary_mask[B,R]. Exactly one valid ego is required, at ANY index.
    Every valid vehicle has complete H13. Masks describe padding, not futures.
    An ego-only mathematical boundary is supported by a real ego-context token;
    the external research-population gate may impose a larger minimum N.

    All normalization is external/shared/FIT-only: history uses channelwise
    population-standard-deviation scale without centering, dimensions use
    their channelwise RMS, and road boundaries use the history-y scale. Each
    scale has a lower bound of one. Relative histories are differences in
    those same normalized coordinates. There is no second internal rescaling.
    Encoder parameters are float32; frozen knots and one joint softmax are
    float64. NaN/Inf padding is sanitized BEFORE every encoder/subtraction.
    """
    VERSION = 'natural_no_focal_ego_scene_cdf_v1'

    def __init__(self, variant, knots, zero_atom_enabled=True, hidden_dim=64, heads=4):
        super().__init__()
        if variant not in ('M0', 'M1', 'M2'):
            raise ValueError('variant must be M0/M1/M2')
        if (type(hidden_dim) is not int or type(heads) is not int
                or hidden_dim < 4 or heads < 1 or hidden_dim % heads):
            raise ValueError('positive hidden width divisible by heads required')
        if (not isinstance(knots, torch.Tensor) or knots.requires_grad
                or knots.dtype not in (torch.float32, torch.float64)
                or type(zero_atom_enabled) is not bool):
            raise ValueError('fixed float32/float64 knots and explicit boolean zero atom required')
        fixed_knots = knots.detach().to(torch.float64).clone()
        self.variant, self.hidden_dim, self.heads = variant, hidden_dim, heads
        self.zero_atom_enabled = zero_atom_enabled
        self.n_outputs = fixed_knots.numel() + int(zero_atom_enabled)
        distribution_from_logits(fixed_knots,
            torch.zeros(self.n_outputs, dtype=torch.float64, device=knots.device),
            zero_atom_enabled=zero_atom_enabled)
        self.vehicle_encoder = _mlp(54, hidden_dim, hidden_dim)
        self.road_boundary_encoder = _mlp(1, hidden_dim, hidden_dim)
        self.road_pool_encoder = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.count_encoder = nn.Linear(1, hidden_dim)
        if variant == 'M0':
            self.pooled_head = nn.Sequential(_mlp(4 * hidden_dim, hidden_dim, hidden_dim),
                                             nn.Linear(hidden_dim, self.n_outputs))
        else:
            self.relative_history_encoder = _mlp(52, hidden_dim, hidden_dim)
            self.pair_encoder = _mlp(4 * hidden_dim, hidden_dim, hidden_dim)
            self.ego_context_encoder = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
            self.range_encoder = nn.Linear(2, hidden_dim)
            self.query_type_embedding = nn.Embedding(3, hidden_dim)
            self.query_encoder = _mlp(3 * hidden_dim, hidden_dim, hidden_dim)
            self.cross_attention = nn.MultiheadAttention(hidden_dim, heads,
                                                         dropout=0., batch_first=True)
            self.interval_scorer = nn.Sequential(_mlp(3 * hidden_dim, hidden_dim, hidden_dim),
                                                 nn.Linear(hidden_dim, 1))
        # Cast modules before registering float64 mathematical buffers.
        self.to(device=knots.device, dtype=torch.float32)
        self.register_buffer('knots', fixed_knots)
        ranges = torch.stack((fixed_knots[:-1], fixed_knots[1:]), -1) / fixed_knots[-1]
        types = torch.zeros(fixed_knots.numel() - 1, dtype=torch.long, device=knots.device)
        if zero_atom_enabled:
            ranges = torch.cat((torch.zeros_like(ranges[:1]), ranges))
            types = torch.cat((torch.ones_like(types[:1]), types))
        ranges = torch.cat((ranges, torch.ones_like(ranges[:1])))
        types = torch.cat((types, torch.full_like(types[:1], 2)))
        self.register_buffer('range_features', ranges.to(torch.float32))
        self.register_buffer('query_types', types)

    def architecture_config(self):
        return dict(architecture_id=self.variant, version=self.VERSION,
            hidden_dim=self.hidden_dim, heads=self.heads,
            parameter_count=sum(p.numel() for p in self.parameters()),
            feature_keys=sorted(FEATURE_KEYS), history_frames=13,
            ego_identification='exactly_one_valid_ego_mask_at_any_index',
            focal_vehicle_input=False, semantics_input=False, actor_ids_input=False,
            future_input=False, actual_pet_input=False, background_role='one_shared_unordered_role',
            encoder_dtype='float32', head_dtype='float64',
            zero_atom_enabled=self.zero_atom_enabled, knots=self.knots.tolist(),
            history_normalization='external_FIT_population_std_scale_only_minimum_scale_1',
            dimensions_normalization='external_FIT_valid_dimensions_RMS_minimum_scale_1',
            road_normalization='external_same_scale_as_history_y',
            internal_static_rescaling=False, count_condition='log1p_valid_agent_count',
            road_encoding='shared_scalar_boundary_MLP_masked_mean_and_max',
            relation_tokens='one_real_ego_context_plus_each_valid_ego_other_pair',
            attention_query='PET_range_dependent' if self.variant == 'M2' else
                            'single_static' if self.variant == 'M1' else 'none',
            PET_range_features_in_scorer=self.variant != 'M0',
            equal_M1_M2_parameters=True, equal_M1_M2_compute_claimed=False,
            pretrained_weights_loaded=False)

    def _clean_inputs(self, features):
        if not isinstance(features, Mapping) or set(features) != FEATURE_KEYS:
            raise ValueError('only the six declared history/static feature keys are accepted')
        if any(not isinstance(value, torch.Tensor) for value in features.values()):
            raise TypeError('feature values must be tensors')
        history, dimensions, road = (features[key] for key in
                                     ('history', 'dimensions', 'road_boundaries'))
        agents, ego, road_mask = (features[key] for key in
                                  ('agent_mask', 'ego_mask', 'road_boundary_mask'))
        if (history.ndim != 4 or history.shape[0] < 1 or history.shape[1] != 13
                or history.shape[2] < 1 or history.shape[3] != 4):
            raise ValueError('history must be nonempty [B,13,N,4]')
        batch, _, count, _ = history.shape
        if (dimensions.shape != (batch, count, 2) or agents.shape != (batch, count)
                or ego.shape != agents.shape or road.ndim != 2 or road.shape[0] != batch
                or road.shape[1] < 2 or road_mask.shape != road.shape):
            raise ValueError('scene static feature/mask shapes disagree')
        if any(value.device != self.knots.device for value in features.values()):
            raise ValueError('all input tensors must share model device')
        if any(value.dtype != torch.float32 for value in (history, dimensions, road)):
            raise ValueError('history/dimensions/road require float32 encoder inputs')
        if any(value.dtype != torch.bool for value in (agents, ego, road_mask)):
            raise ValueError('all masks must be boolean')
        if (bool((ego & ~agents).any()) or not bool((ego.sum(1) == 1).all())
                or not bool((road_mask.sum(1) >= 2).all())):
            raise ValueError('exactly one valid ego and at least two observed road boundaries required')
        history = torch.where(agents[:, None, :, None], history, torch.zeros_like(history))
        dimensions = torch.where(agents[..., None], dimensions, torch.zeros_like(dimensions))
        road = torch.where(road_mask, road, torch.zeros_like(road))
        if (not bool(torch.isfinite(history).all()) or not bool(torch.isfinite(dimensions).all())
                or not bool(torch.isfinite(road).all()) or bool((dimensions[agents] <= 0).any())):
            raise ValueError('valid observations/static inputs must be finite and dimensions positive')
        return history, dimensions, road, agents, ego, road_mask

    def _encode(self, features):
        history, dimensions, road, mask, ego_mask, road_mask = self._clean_inputs(features)
        batch, _, count, _ = history.shape
        temporal = history.permute(0, 2, 1, 3).reshape(batch, count, 52)
        vehicle = self.vehicle_encoder(torch.cat((temporal, dimensions), -1))
        vehicle = torch.where(mask[..., None], vehicle, torch.zeros_like(vehicle))
        road_tokens = self.road_boundary_encoder(road[..., None])
        road_mean, road_max = _masked_pool(road_tokens, road_mask)
        context = self.road_pool_encoder(torch.cat((road_mean, road_max), -1))
        context = context + self.count_encoder(torch.log1p(mask.sum(1).to(history.dtype))[:, None])
        ego = torch.where(ego_mask[..., None], vehicle, torch.zeros_like(vehicle)).sum(1)
        return history, vehicle, ego, context, mask, ego_mask

    def interaction_readout(self, features):
        """Inspect no-focal relation masks/attention; not a PET-label input path."""
        if self.variant == 'M0':
            raise ValueError('M0 has no relation attention')
        history, vehicle, ego, context, mask, ego_mask = self._encode(features)
        batch, _, count, _ = history.shape
        other_mask = mask & ~ego_mask
        ego_history = torch.where(ego_mask[:, None, :, None], history, torch.zeros_like(history)).sum(2)
        relative = torch.where(other_mask[:, None, :, None], history - ego_history[:, :, None],
                               torch.zeros_like(history))
        delta = self.relative_history_encoder(relative.permute(0, 2, 1, 3).reshape(batch, count, 52))
        pair = self.pair_encoder(torch.cat((ego[:, None].expand(-1, count, -1), vehicle,
                                            delta, context[:, None].expand(-1, count, -1)), -1))
        pair = torch.where(other_mask[..., None], pair, torch.zeros_like(pair))
        anchor = self.ego_context_encoder(torch.cat((ego, context), -1))
        tokens = torch.cat((anchor[:, None], pair), 1)
        token_mask = torch.cat((torch.ones((batch, 1), dtype=torch.bool, device=mask.device), other_mask), 1)
        ranges = self.range_encoder(self.range_features) + self.query_type_embedding(self.query_types)
        repeated_ego = ego[:, None].expand(-1, self.n_outputs, -1)
        queries = self.query_encoder(torch.cat((repeated_ego,
            context[:, None].expand(-1, self.n_outputs, -1), ranges[None].expand(batch, -1, -1)), -1))
        if self.variant == 'M2':
            attention_query = queries
        else:
            attention_query = self.query_encoder(torch.cat((ego, context, torch.zeros_like(context)), -1))[:, None]
        values, weights = self.cross_attention(attention_query, tokens, tokens,
            key_padding_mask=~token_mask, need_weights=True, average_attn_weights=False)
        if self.variant == 'M1':
            values = values.expand(-1, self.n_outputs, -1)
            weights = weights.expand(-1, -1, self.n_outputs, -1)
        logits = self.interval_scorer(torch.cat((repeated_ego, queries, values), -1)).squeeze(-1)
        return dict(logits=logits, attention_weights=weights, readout_values=values,
                    relation_tokens=tokens, relation_mask=token_mask,
                    attention_query_count=attention_query.shape[1])

    def forward(self, features):
        if self.variant == 'M0':
            _history, vehicle, ego, context, mask, _ego_mask = self._encode(features)
            mean, maximum = _masked_pool(vehicle, mask)
            logits = self.pooled_head(torch.cat((ego, mean, maximum, context), -1))
        else:
            logits = self.interaction_readout(features)['logits']
        return distribution_from_logits(self.knots, logits.to(torch.float64),
            zero_atom_enabled=self.zero_atom_enabled, model_version=self.VERSION + ':' + self.variant)

    predict_distribution = forward


def count_scene_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())
