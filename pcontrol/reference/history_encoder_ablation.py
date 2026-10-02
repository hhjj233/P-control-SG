"""Bounded history-encoder/readout controls; frozen historical sources untouched."""
import torch
from torch import nn

from .scene_models import SceneCDFReference
from .time_attention_cdf import ObservedHistoryEncoder, TimeAttentionSceneCDF

NEW_ARMS=('M2_TimeAttn_absolute_only','M2_TimeAttn_relative_only','M1_TimeAttn','M2_GRU')


class ObservedGRUEncoder(nn.Module):
    """Two-layer observed-only GRU, shared over actors; last hidden-state pooling."""
    def __init__(self,width=64,layers=2,with_dimensions=False):
        super().__init__();self.input_dim=54 if with_dimensions else 52
        self.state_projection=nn.Linear(4,width)
        self.gru=nn.GRU(width,width,num_layers=layers,batch_first=True,dropout=0.)
        self.dimension_projection=nn.Linear(2,width) if with_dimensions else None
        self.output_norm=nn.LayerNorm(width)

    def forward(self,inputs):
        if inputs.ndim<2 or inputs.shape[-1]!=self.input_dim:raise ValueError('observed H13x4, optionally dimensions, required')
        prefix=inputs.shape[:-1];flat=inputs.reshape(-1,self.input_dim)
        sequence=self.state_projection(flat[:,:52].reshape(-1,13,4))
        _,hidden=self.gru(sequence);pooled=hidden[-1]
        if self.dimension_projection is not None:pooled=pooled+self.dimension_projection(flat[:,52:])
        return self.output_norm(pooled).reshape(*prefix,-1)


class HistoryEncoderAblation(SceneCDFReference):
    VERSION='natural_history_encoder_publication_ablation_v1'

    def __init__(self,arm,knots,*,hidden_dim=64,heads=4,temporal_layers=2,temporal_feedforward_dim=128):
        if arm not in NEW_ARMS:raise ValueError('undeclared ablation')
        super().__init__('M1' if arm=='M1_TimeAttn' else 'M2',knots,hidden_dim=hidden_dim,heads=heads)
        self.arm=arm;self.temporal_layers=temporal_layers;self.temporal_feedforward_dim=temporal_feedforward_dim
        if arm=='M2_GRU':
            self.vehicle_encoder=ObservedGRUEncoder(hidden_dim,temporal_layers,True)
            self.relative_history_encoder=ObservedGRUEncoder(hidden_dim,temporal_layers,False)
        else:
            # Always initialize BOTH in the full-model order, even when one is
            # discarded. Retained temporal weights then match the full branch;
            # retained MLP weights match its same-seed scratch MLP comparator.
            vehicle=ObservedHistoryEncoder(hidden_dim,heads,temporal_layers,temporal_feedforward_dim,True)
            relative=ObservedHistoryEncoder(hidden_dim,heads,temporal_layers,temporal_feedforward_dim,False)
            if arm!='M2_TimeAttn_relative_only':self.vehicle_encoder=vehicle
            if arm!='M2_TimeAttn_absolute_only':self.relative_history_encoder=relative
        self.vehicle_encoder.to(device=self.knots.device,dtype=torch.float32)
        self.relative_history_encoder.to(device=self.knots.device,dtype=torch.float32)

    def architecture_config(self):
        cfg=super().architecture_config()
        cfg.update(architecture_id=self.arm,version=self.VERSION,temporal_layers=self.temporal_layers,
            temporal_feedforward_dim=self.temporal_feedforward_dim,
            absolute_history_encoder=type(self.vehicle_encoder).__name__,
            relative_history_encoder=type(self.relative_history_encoder).__name__,
            temporal_dropout=0.,future_tokens=False,pretrained_weights_loaded=False,
            equal_parameters_or_FLOPs_claimed=False,original_mixed_head_and_CRPS_preserved=True)
        return cfg


def make_ablation(arm,seed=20260915,hidden_dim=64):
    torch.manual_seed(seed);knots=torch.linspace(0,4,65,dtype=torch.float64)
    if arm=='M2_MLP':return SceneCDFReference('M2',knots,hidden_dim=hidden_dim,heads=4)
    if arm=='M2_TimeAttn':return TimeAttentionSceneCDF(knots,hidden_dim=hidden_dim,heads=4,temporal_feedforward_dim=2*hidden_dim)
    return HistoryEncoderAblation(arm,knots,hidden_dim=hidden_dim,heads=4,temporal_feedforward_dim=2*hidden_dim)
