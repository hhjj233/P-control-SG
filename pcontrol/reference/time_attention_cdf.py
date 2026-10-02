"""M2 with actual temporal encoders in place of BOTH flattened-history MLPs.

Only the vehicle and relative-history encoder modules are replaced. The frozen
SceneCDFReference input gate, relation construction, PET-range queries, road
encoding, mixed-CDF head, and proper scoring interface are inherited unchanged.
This is a from-scratch candidate, NOT a zero-output residual warm start.

Conceptual reference: HiVT's temporal attention encoder (CVPR 2022). This small
implementation is written locally using PyTorch primitives; no third-party
weights, prediction head, dataset adapter, or future encoder is imported.
"""
import torch
from torch import nn

from .scene_models import SceneCDFReference


class TemporalAttentionBlock(nn.Module):
    """Pre-normalized self-attention over already observed time steps only."""

    def __init__(self, width, heads, feedforward_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, feedforward_dim), nn.GELU(),
                                         nn.Linear(feedforward_dim, width))

    def forward(self, value):
        normed = self.norm1(value)
        # An explicit small attention matrix avoids a backend-dependent fused
        # attention path. 13 observed tokens + one summary token per actor.
        attended, _ = self.attention(normed, normed, normed, need_weights=True,
                                     average_attn_weights=False)
        value = value+attended
        return value+self.feedforward(self.norm2(value))


class ObservedHistoryEncoder(nn.Module):
    """Shared temporal operator for flattened input at the unchanged M2 boundary.

    The flattening is undone immediately: [*,13*4(+2)] -> [*,13,4]. The last
    two scalars, when enabled, are actual vehicle dimensions and are fused only
    after temporal aggregation. Actor and relative streams have distinct learned
    encoders but the same architecture. No actor index is embedded.
    """

    def __init__(self, width=64, heads=4, layers=2, feedforward_dim=128, with_dimensions=False):
        super().__init__()
        if (type(width) is not int or type(heads) is not int or width < 4 or heads < 1 or width % heads
                or type(layers) is not int or layers < 1 or type(feedforward_dim) is not int
                or feedforward_dim < width or type(with_dimensions) is not bool):
            raise ValueError("valid temporal width, heads, layers, FF width, and dimension flag required")
        self.width = width; self.with_dimensions = with_dimensions
        self.input_dim = 54 if with_dimensions else 52
        self.state_projection = nn.Linear(4, width)
        self.summary_token = nn.Parameter(torch.empty(1, 1, width))
        self.time_embedding = nn.Parameter(torch.empty(1, 14, width))
        self.blocks = nn.ModuleList([TemporalAttentionBlock(width, heads, feedforward_dim) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(width)
        self.dimension_projection = nn.Linear(2, width) if with_dimensions else None
        self.register_buffer("observed_times_seconds", torch.linspace(-.96, 0., 13))
        nn.init.normal_(self.summary_token, std=.02)
        nn.init.normal_(self.time_embedding, std=.02)

    def forward(self, inputs):
        if inputs.ndim < 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError("expected observed H13x4, optionally followed by two size scalars")
        prefix = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.input_dim)
        history = flat[:, :52].reshape(-1, 13, 4)
        tokens = self.state_projection(history)
        tokens = torch.cat((tokens, self.summary_token.expand(len(flat), -1, -1)), dim=1)
        tokens = tokens+self.time_embedding
        for block in self.blocks:
            tokens = block(tokens)
        pooled = tokens[:, -1]
        if self.with_dimensions:
            pooled = pooled+self.dimension_projection(flat[:, 52:])
        return self.output_norm(pooled).reshape(*prefix, self.width)


class TimeAttentionSceneCDF(SceneCDFReference):
    VERSION = "natural_M2_two_history_time_attention_CDF_v1"

    def __init__(self, knots, *, zero_atom_enabled=True, hidden_dim=64, heads=4,
                 temporal_layers=2, temporal_feedforward_dim=128):
        # Parent construction is deliberately identical to M2 before replacing
        # the encoders: a same-seed scratch control can share all other initial
        # tensors exactly. The original two MLPs are removed, not bypassed.
        super().__init__("M2", knots, zero_atom_enabled=zero_atom_enabled,
                         hidden_dim=hidden_dim, heads=heads)
        self.temporal_layers = temporal_layers
        self.temporal_feedforward_dim = temporal_feedforward_dim
        self.vehicle_encoder = ObservedHistoryEncoder(hidden_dim, heads, temporal_layers,
                                                      temporal_feedforward_dim, with_dimensions=True)
        self.relative_history_encoder = ObservedHistoryEncoder(hidden_dim, heads, temporal_layers,
                                                               temporal_feedforward_dim, with_dimensions=False)
        self.vehicle_encoder.to(device=self.knots.device, dtype=torch.float32)
        self.relative_history_encoder.to(device=self.knots.device, dtype=torch.float32)

    def architecture_config(self):
        config = super().architecture_config()
        config.update(architecture_id="M2_TimeAttn", version=self.VERSION,
            history_encoding="two_separate_shared_actorwise_time_Transformers_instead_of_flat_MLPs",
            replaced_modules=["vehicle_encoder", "relative_history_encoder"],
            temporal_layers=self.temporal_layers, temporal_heads=self.heads,
            temporal_feedforward_dim=self.temporal_feedforward_dim,
            temporal_tokens=13, temporal_summary_tokens=1,
            temporal_position_encoding="learned_14_positions; H timestamps -0.96:0.08:0 seconds",
            temporal_attention_scope="within_one_observed_history_only; no_future_tokens",
            temporal_causal_mask=False, temporal_dropout=0.,
            actor_and_relative_temporal_parameters_shared=False,
            encoder_shared_across_all_valid_vehicles=True,
            new_cross_vehicle_attention=False, residual_around_old_MLP=False,
            original_range_relation_head_preserved=True,
            from_scratch=True, pretrained_weights_loaded=False,
            equal_M1_M2_parameters=False, equal_M1_M2_compute_claimed=False)
        return config
