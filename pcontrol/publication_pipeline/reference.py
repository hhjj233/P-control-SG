"""Validated frozen reference for the group-protected publication branch."""
import copy

import torch

from pcontrol.time_attention_pipeline import common as c
from pcontrol.plugins.risk_plugin import FrozenRiskPlugin
from pcontrol.research import train_natural_scene_reference as io

PROTOCOL='guarded_time_attention_reference_v1'


class GuardedTimeAttentionRiskPlugin(FrozenRiskPlugin):
    @classmethod
    def from_manifest(cls,binding,device='cpu'):
        m=c.json_file(binding)
        if (m['protocol']!=PROTOCOL or m['status']!='complete' or m['architecture']!='M2_TimeAttn'
                or m['base_selection']!='STOP_group_guarded' or m['calibration_selected_on']!='CAL_record_LOO_only'):
            raise ValueError('complete guarded, CAL-only reference required')
        c.verify_sources(m['code_sha256'])
        cp=torch.load(io.verify_binding(m['checkpoint']),map_location='cpu',weights_only=False)
        if cp['arm']!='M2_TimeAttn' or cp['normalizer']!=m['normalizer'] or cp['data']!=m['data']:
            raise ValueError('reference lineage mismatch')
        with torch.random.fork_rng(devices=[]):model=c.make_reference_model()
        model.load_state_dict(cp['state_dict'],strict=True);model.to(device)
        warp=c.StableCountWarp.from_dict(c.json_file(m['calibration_model']))
        result=cls(model,c.json_file(m['normalizer']),warp,provenance=dict(reference_manifest=copy.deepcopy(binding),
            checkpoint=m['checkpoint'],normalizer=m['normalizer'],calibration_model=m['calibration_model']))
        result._metadata.update(protocol='guarded_time_attention_risk_plugin_v1',estimator='M2_TimeAttn',
            base_selection='STOP_group_guarded',pipeline_bound=True,production_bound=False,
            reference_version=binding['sha256'],risk_CDF_is_estimated_not_known_true=True,
            validation_reuse=m['validation_reuse'],model_training=False)
        return result
