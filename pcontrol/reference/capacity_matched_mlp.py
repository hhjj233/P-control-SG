"""Nearly equal-parameter flat-history control, chosen without data fitting."""
import torch
from .scene_models import SceneCDFReference,_mlp


class CapacityMatchedMLPCDF(SceneCDFReference):
    VERSION='natural_M2_near_capacity_matched_flat_history_MLP_v1'
    ARM='M2_WideMLP'

    def __init__(self,knots,*,encoder_hidden_width=578):
        if type(encoder_hidden_width) is not int or encoder_hidden_width<1:raise ValueError('positive MLP width required')
        # Match the same-seed nonencoder initialization of all prior arms.
        super().__init__('M2',knots,hidden_dim=64,heads=4)
        self.encoder_hidden_width=encoder_hidden_width
        self.vehicle_encoder=_mlp(54,encoder_hidden_width,64)
        self.relative_history_encoder=_mlp(52,encoder_hidden_width,64)
        self.vehicle_encoder.to(device=self.knots.device,dtype=torch.float32)
        self.relative_history_encoder.to(device=self.knots.device,dtype=torch.float32)

    def architecture_config(self):
        config=super().architecture_config()
        config.update(architecture_id=self.ARM,version=self.VERSION,encoder_hidden_width=self.encoder_hidden_width,
            absolute_history_encoder='flat_H13x4_and_dimensions_two_layer_MLP',relative_history_encoder='flat_relative_H13x4_two_layer_MLP',
            encoder_width_chosen_from_parameter_counts_only=True,temporal_attention=False,pretrained_weights_loaded=False,
            equal_FLOPs_claimed=False,exact_equal_parameters_claimed=False)
        return config


def make_capacity_control(seed=20260915,width=578):
    torch.manual_seed(seed)
    return CapacityMatchedMLPCDF(torch.linspace(0,4,65,dtype=torch.float64),encoder_hidden_width=width)
