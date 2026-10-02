"""Read-only loading of the seven preselected references for final evaluation.

Loading this adapter reads only known model/calibration artifacts. It never
opens a traffic dataset; the separate protected-data gate remains mandatory.
"""
import copy
import numpy as np
import torch
from pcontrol.publication_pipeline.final_validation_protocol import (
    PROTOCOL,REFERENCE_ARMS,verified_json,same_binding,digest,resolve)
from pcontrol.reference.history_encoder_ablation import make_ablation
from pcontrol.reference.capacity_matched_mlp import make_capacity_control
from pcontrol.plugins.risk_plugin import FrozenRiskPlugin
from pcontrol.time_attention_pipeline import common as c
from pcontrol.research.pilot_time_attention_cdf import tensor_hash

FEATURES = {'history','dimensions','road_boundaries','ego_mask','agent_mask'}


class FrozenFinalReference(FrozenRiskPlugin):
    @classmethod
    def from_contract(cls, contract_binding, arm, *, view='frozen_calibrated', device='cpu'):
        contract = verified_json(contract_binding)
        if contract['protocol'] != PROTOCOL or arm not in REFERENCE_ARMS:
            raise ValueError('known frozen final reference required')
        if view not in ('raw','frozen_calibrated'): raise ValueError('no final-data calibration fitting')
        c.verify_sources(contract['inherited_generator_code_sha256'])
        record = contract['reference_models'][arm];descriptor=record['descriptor']
        binding = descriptor['checkpoint']
        if digest(binding['path']) != binding['sha256']: raise ValueError('checkpoint drift')
        cp = torch.load(resolve(binding['path']),map_location='cpu',weights_only=False)
        for key in ('normalizer','data'):
            if not same_binding(cp[key],descriptor[key]): raise ValueError('reference training lineage drift')
        expected_state = descriptor.get('state_tensor_sha256',descriptor.get('state_sha256'))
        if expected_state is None or tensor_hash(cp['state_dict']) != expected_state:
            raise ValueError('frozen neural state differs')
        # Initializers are overwritten by saved weights and may not consume
        # the independently controlled generation RNG stream.
        with torch.random.fork_rng(devices=[]):
            if arm == 'M2_WideMLP':
                model=make_capacity_control(width=descriptor['architecture']['encoder_hidden_width'])
            else: model=make_ablation(arm)
        if model.architecture_config() != descriptor['architecture']:
            raise ValueError('reference architecture differs from selected model')
        model.load_state_dict(cp['state_dict'],strict=True);model.to(device)
        warp = c.StableCountWarp.identity() if view=='raw' else c.StableCountWarp.from_dict(verified_json(record['calibration_model']))
        plugin=cls(model,verified_json(descriptor['normalizer']),warp,provenance=dict(
            final_model_contract=copy.deepcopy(contract_binding),checkpoint=copy.deepcopy(binding),
            normalizer=copy.deepcopy(descriptor['normalizer']),calibration_model=copy.deepcopy(record['calibration_model']),view=view))
        plugin._metadata.update(estimator=arm,evaluation_scope='frozen_reference_interface_no_evaluation_population_claim',
            final_data_decoded_by_constructor=False,production_bound=False,checkpoint_and_calibration_frozen=True,
            reference_view=view,model_training=False)
        return plugin

    def condition_features(self, features):
        if set(features) != FEATURES: raise ValueError('history/static features only; no identities, labels or futures')
        previous=torch.backends.mha.get_fastpath_enabled()
        try:
            # Match P5's canonical CPU reference conditioning. The caller's
            # generator attention backend is restored immediately afterwards.
            if next(self._model.parameters()).device.type=='cpu':torch.backends.mha.set_fastpath_enabled(True)
            return self.condition(*(features[k] for k in ('history','dimensions','road_boundaries','ego_mask','agent_mask')))
        finally:
            torch.backends.mha.set_fastpath_enabled(previous)


def check_reference_values(reference):
    grid=np.linspace(0.,4.,257)
    left=np.asarray(reference.cdf(grid,side='left'));right=np.asarray(reference.cdf(grid,side='right'))
    if (not np.isfinite(left).all() or not np.isfinite(right).all() or np.any(left>right+1e-12)
            or np.any(np.diff(right)<-1e-12) or left[0]!=0. or right[-1]!=1.):
        raise ValueError('invalid mixed CDF values')
    return dict(num_agents=reference.num_agents,zero_mass=float(right[0]),cap_mass=float(1-left[-1]),
        cdf_monotone=True,endpoint_limits_valid=True,grid_points=len(grid))
