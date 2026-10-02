"""Explicit version bindings and pure distribution adapters for the new pipeline."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pcontrol.research import train_natural_scene_reference as io
from pcontrol.reference.time_attention_cdf import TimeAttentionSceneCDF
from pcontrol.reference.scene_calibration import SceneCDFWarp, _physical_pieces
from pcontrol.reference.scene_context_plugin import SceneContextCDFPlugin
from pcontrol.reference.torch_frozen_inverse import FrozenTorchInverseCDF
from pcontrol.plugins.risk_plugin import FrozenRiskPlugin

ROOT=Path(__file__).resolve().parents[2]
PIPELINE="natural_time_attention_full_pipeline_v1"
OUTPUT=ROOT/"outputs/natural_percentile/time_attention_full_pipeline_v1_20260915"


def bind(path):
    p=io.resolve(path)
    return dict(path=str(p),sha256=io.sha256(p))


def json_file(binding):
    return json.loads(io.verify_binding(binding).read_text())


def arrays(binding):
    with np.load(io.verify_binding(binding),allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def source_bindings(names):
    return {name:io.sha256(ROOT/name) for name in names}


def verify_sources(expected):
    for name,digest in expected.items():
        if io.sha256(ROOT/name)!=digest:raise ValueError("frozen source changed: "+name)


def policy(binding):
    p=json_file(binding)
    if (p["protocol"]!=PIPELINE or p["only_natural_observations"] is not True
            or p["P_condition_guidance_ablation"] is not False
            or p["reference_architecture"]!="M2_TimeAttn"
            or p["output_root"]!="outputs/natural_percentile/time_attention_full_pipeline_v1_20260915"):
        raise ValueError("wrong full pipeline scope")
    return p


def count_context(counts):
    n=np.asarray(counts,dtype=np.float64).reshape(-1)
    return np.column_stack((n,np.zeros((len(n),3))))


class StableCountWarp(SceneCDFWarp):
    """Same global/count model, with the versioned stable mixed-CDF arithmetic.

    Remaining context columns are unused by global/count families. They are
    placeholders, not inferred speed/gap features and never learned inputs.
    """
    def _stable(self):return SceneContextCDFPlugin(self.family,self.node_values)

    def row_nodes(self,counts):return self._stable().row_nodes(count_context(counts))

    def cdf(self,masses,counts,y,*,side="right",base_knots=None):
        return self._stable().cdf(masses,count_context(counts),y,side=side,base_knots=base_knots)

    def quantile(self,masses,counts,u,*,base_knots=None):
        return self._stable().quantile(masses,count_context(counts),u,base_knots=base_knots)

    def rank(self,masses,counts,y,*,base_knots=None):
        return self._stable().rank(masses,count_context(counts),y,base_knots=base_knots)

    def crps(self,masses,counts,y,*,base_knots=None,normalized=False):
        value=self._stable().crps(masses,count_context(counts),y,base_knots=base_knots)
        cap=4. if base_knots is None else float(base_knots[-1])
        return value/cap if normalized else value


class StableTorchInverse(FrozenTorchInverseCDF):
    """Canonical effective nodes and identical endpoint reduction to NumPy."""
    VERSION="time_attention_pipeline_stable_mixed_inverse_v1"

    def __init__(self,masses,counts,warp=None,base_knots=None,*,row_nodes=None):
        if warp is not None:
            if row_nodes is not None:raise ValueError("one warp or explicit row nodes")
            counts_np=counts.detach().cpu().numpy() if isinstance(counts,torch.Tensor) else np.asarray(counts)
            row_nodes=warp.row_nodes(counts_np)
        super().__init__(masses,counts,warp=None,base_knots=base_knots,row_nodes=row_nodes)
        mass=self.masses.cpu().numpy();knots=self.base_knots.cpu().numpy();u=self.u_knots.cpu().numpy()
        nodes=self.row_nodes.cpu().numpy();left=self.inverse_left.cpu().numpy().copy();right=self.inverse_right.cpu().numpy().copy()
        for i in range(len(mass)):
            physical,lb,rb=_physical_pieces(mass[i],knots,u);n=len(physical)
            left[i,:n]=np.sum(lb*nodes[i],axis=-1);right[i,:n]=np.sum(rb*nodes[i],axis=-1)
        self.inverse_left.copy_(torch.as_tensor(left,device=self.masses.device))
        self.inverse_right.copy_(torch.as_tensor(right,device=self.masses.device))


def make_reference_model():
    return TimeAttentionSceneCDF(torch.linspace(0.,4.,65,dtype=torch.float64),hidden_dim=64,heads=4,
                                 temporal_layers=2,temporal_feedforward_dim=128)


class TimeAttentionRiskPlugin(FrozenRiskPlugin):
    """Hash-bound full-FIT transformer reference for the complete new workflow."""

    @classmethod
    def from_manifest(cls,binding,device="cpu"):
        m=json_file(binding)
        if (m["protocol"]!="time_attention_reference_validation_v1" or m["status"]!="complete"
                or m["architecture"]!="M2_TimeAttn" or m["calibration_selected_on"]!="CAL_record_LOO_only"):
            raise ValueError("complete validated Transformer reference manifest required")
        verify_sources(m["code_sha256"])
        checkpoint=torch.load(io.verify_binding(m["checkpoint"]),map_location="cpu",weights_only=False)
        if checkpoint["arm"]!="M2_TimeAttn" or checkpoint["normalizer"]!=m["normalizer"]:
            raise ValueError("reference checkpoint lineage mismatch")
        with torch.random.fork_rng(devices=[]):model=make_reference_model()
        model.load_state_dict(checkpoint["state_dict"],strict=True);model.to(device)
        warp=StableCountWarp.from_dict(json_file(m["calibration_model"]))
        result=cls(model,json_file(m["normalizer"]),warp,provenance=dict(reference_manifest=copy.deepcopy(binding),
                   checkpoint=m["checkpoint"],calibration_model=m["calibration_model"],normalizer=m["normalizer"]))
        result._metadata.update(protocol="frozen_time_attention_risk_plugin_v1",estimator="M2_TimeAttn",
            pipeline_bound=True,production_bound=False,model_training=False,
            calibration_status="evaluated_estimated_reference_not_true_conditional_CDF",
            reference_version=binding["sha256"],numerical_adapter="StableCountWarp_v1")
        return result
