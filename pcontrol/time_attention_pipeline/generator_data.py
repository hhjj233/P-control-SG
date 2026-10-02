"""New Transformer label/context joins and risk cache; natural geometry reused."""
from collections import Counter
from functools import lru_cache

import numpy as np
import torch

from . import common as c
from pcontrol.research import train_natural_scene_reference as io
from pcontrol.research import train_natural_diffusion as prior
from pcontrol.research import train_natural_direct_p as direct
from pcontrol.research import train_natural_risk_tangent as tangent_training
from pcontrol.reference.direct_p_crossfit import _load_pack
from pcontrol.reference.torch_frozen_cdf import FrozenTorchCDF
from pcontrol.reference.torch_frozen_inverse import PIECES_KEY,quantile_from_pieces
from pcontrol.generation.cdf_shape_context import CONTEXT_KEY
from pcontrol.generation.risk_guidance import TorchTrajectoryDecoder
from pcontrol.generation.trajectory_basis import TrajectoryBasis
from pcontrol.generation.broadphase_geometry import BroadphaseGeometryAdapter
from pcontrol.generation import risk_tangent

OLD_GEOMETRY=dict(path="outputs/natural_percentile/natural_risk_tangent_cache_v1_20260910/manifest.json",
                  sha256="91393995f81419ca620065a38d5def6aa239f31d984bb2921e2668221a834f2b")
PROTOCOL="time_attention_generator_data_v1"


def join_rows(pack,source):
    fields=("scene_id","recording_id","role")
    a=list(zip(*(pack[k].astype(str).tolist() for k in fields)))
    b=list(zip(*(source[k].astype(str).tolist() for k in fields)))
    if len(a)!=len(set(a)) or len(b)!=len(set(b)) or set(a)!=set(b):raise ValueError("exact scene/recording/role join required")
    lookup={key:i for i,key in enumerate(b)};order=np.array([lookup[key] for key in a])
    return {k:v[order] for k,v in source.items()}


def load_inputs(pb):
    p=c.policy(pb);labels_binding=c.bind(c.OUTPUT/"crossfit/manifest.json");labels_manifest=c.json_file(labels_binding)
    if labels_manifest["status"]!="complete" or labels_manifest["policy"]!=pb:raise ValueError("complete new Transformer OOF labels required")
    c.verify_sources(labels_manifest["code_sha256"])
    data=c.json_file(p["generator_data"]);data_policy=c.json_file(data["policy"])
    prior.validate_prepared_metadata(data,data_policy,data["policy"])
    packs={};labels={};contexts={};pvalues={};joins={}
    for role in ("FIT","STOP"):
        pack=prior.load_pack(data["packs"][role],role)
        label=join_rows(pack,c.arrays(labels_manifest["roles"][role]["labels"]))
        context=join_rows(pack,c.arrays(labels_manifest["roles"][role]["context"]))
        values,evidence=direct.join_percentile_labels(pack,label,role=role)
        if (not np.array_equal(context["pet_seconds"],label["pet_seconds"])
                or not np.array_equal(context["num_agents"],pack["agent_mask"].sum(1))
                or not np.array_equal(context["fold"],label["fold"])):
            raise ValueError("label and conditioning distributions are not the same histories/folds")
        ref=FrozenTorchCDF(context["joint_masses"],context["num_agents"],row_nodes=context["row_nodes"])
        replay=ref.rank(torch.from_numpy(label["pet_seconds"]))["p_mid"].numpy()
        error=float(np.max(abs(replay-values)))
        if error>1e-12:raise ValueError("new labels do not replay under their own CDFs")
        pieces=torch.tensor(context[PIECES_KEY]);q=quantile_from_pieces(pieces,torch.tensor(1.-values))
        inverse_replay=float(np.max(abs(ref.rank(q)["p_mid"].numpy()-values)))
        if inverse_replay>1e-10:raise ValueError("natural midpoint labels fail exact target inverse consistency")
        evidence.update(own_CDF_replay_max_error=error,natural_label_inverse_replay_max_error=inverse_replay)
        packs[role],labels[role],contexts[role],pvalues[role],joins[role]=pack,label,context,values,evidence
    if set(packs["FIT"]["recording_id"])&set(packs["STOP"]["recording_id"]):raise ValueError("FIT/STOP recording overlap")
    physical=join_rows(packs["FIT"],_load_pack(labels_manifest["physical_packs"]["FIT"],"FIT"))
    if not np.array_equal(physical["target"],labels["FIT"]["pet_seconds"]):raise ValueError("natural PET provenance changed")
    mask=packs["FIT"]["agent_mask"]
    if not np.array_equal(physical["agent_mask"],mask) or not np.array_equal(physical["history"][:,-1][mask],packs["FIT"]["anchors"][mask]):
        raise ValueError("actual roster or physical anchors changed")
    return dict(policy=p,data=data,packs=packs,labels=labels,contexts=contexts,pvalues=pvalues,joins=joins,
                physical=physical,labels_manifest=labels_manifest,labels_binding=labels_binding)


def regenerate_tangent(source):
    old=c.json_file(OLD_GEOMETRY);old_policy=c.json_file(old["policy"])
    p=source["policy"];data=source["data"];pack=source["packs"]["FIT"]
    if (old["generator_data"]["sha256"]!=p["generator_data"]["sha256"]
            or old["coefficient_normalizer"]["sha256"]!=data["coefficient_normalizer"]["sha256"]
            or old["h_definition"]!="h=f*g=dCDF_dc" or old["risk_rank_tangent_sign"]!=-1):
        raise ValueError("only unchanged physical PET geometry may be reused")
    c.verify_sources(old["code_sha256"])
    previous=join_rows(pack,c.arrays(old["FIT_array"]))
    labels=source["labels"]["FIT"];ctx=source["contexts"]["FIT"]
    if not np.array_equal(previous["originalPET"],labels["pet_seconds"]) or not np.array_equal(previous["agent_mask"],pack["agent_mask"]):
        raise ValueError("cached geometry belongs to different real observations")
    result={k:v.copy() for k,v in previous.items()}
    result["p_mid"]=labels["p_mid"].copy();result["fold"]=labels["fold"].copy()
    for key in ("density","density_candidate","density_left","density_right","density_relative_difference","cdf_reconstruction_difference"):
        result[key].fill(0.)
    result["h"].fill(0.);result["valid_density"].fill(False);result["density_reason"][:]='not_selected'
    cfg=old_policy["cache"]
    for i in np.flatnonzero(result["selected_subset_mask"]):
        warp=c.StableCountWarp("global",ctx["row_nodes"][i:i+1])
        d=risk_tangent.oof_density(warp,ctx["joint_masses"][i],int(ctx["num_agents"][i]),
            result["originalPET"][i],result["decodedPET"][i],epsilon=cfg["density_difference_step_seconds"],
            relative_tolerance=cfg["density_left_right_relative_tolerance"],minimum_density=cfg["density_minimum"],
            cdf_difference_limit=cfg["cdf_reconstruction_difference_max"])
        result["valid_density"][i]=d["valid_density"];result["density_reason"][i]=d["density_reason"]
        for key in ("density_left","density_right","density_relative_difference","cdf_reconstruction_difference"):result[key][i]=d[key]
        result["density_candidate"][i]=d["density"];result["density"][i]=d["density"] if d["valid_density"] else 0.
        result["h"][i]=result["density"][i]*result["g"][i]
    summary=dict(rows=len(labels["p_mid"]),selected_count=int(result["selected_subset_mask"].sum()),
                 valid_geom=int(result["valid_geom"].sum()),valid_density=int(result["valid_density"].sum()),
                 valid_jac=int((result["valid_geom"]&result["valid_density"]).sum()))
    risk_tangent._validate_cache_arrays(result,summary)
    tangent_training.join_risk_cache(pack,source["pvalues"]["FIT"],result)
    summary.update(old_geometry_manifest=OLD_GEOMETRY,old_geometry_arrays=old["FIT_array"],
        geometry_bitwise_unchanged=all(np.array_equal(result[k],previous[k]) for k in ("g","unit_g","valid_geom","decodedPET","gradient_norm")),
        old_density_or_rank_gradient_reused=False,new_CDF_context=source["labels_manifest"]["roles"]["FIT"]["context"],
        density_reasons=dict(Counter(result["density_reason"][result["selected_subset_mask"]])))
    return result,summary


class TransformerFITTeacher:
    """New per-row OOF distributions, same full-trajectory geometry objectives."""
    def __init__(self,source,device):
        self.pack=source["packs"]["FIT"];self.physical=source["physical"];self.device=device
        self.masses=source["contexts"]["FIT"]["joint_masses"];self.nodes=source["contexts"]["FIT"]["row_nodes"]
        self.normalizer=c.json_file(source["data"]["coefficient_normalizer"])
        self.replay_max_error=source["joins"]["FIT"]["own_CDF_replay_max_error"]
        if not self.pack["ego_mask"][:,0].all() or not np.all(self.pack["ego_mask"].sum(1)==1):
            raise ValueError("training geometry requires the unchanged first-ego stored roster")

    @lru_cache(maxsize=512)
    def geometry(self,row):
        mask=self.pack["agent_mask"][row];anchors=self.pack["anchors"][row,mask]
        return (TorchTrajectoryDecoder(TrajectoryBasis(8),self.normalizer,anchors,device=self.device),
                BroadphaseGeometryAdapter(self.physical["dimensions"][row,mask],anchors))

    def batch(self,rows):
        rows=np.asarray(rows,dtype=np.int64);counts=self.pack["agent_mask"][rows].sum(1)
        reference=FrozenTorchCDF(torch.tensor(self.masses[rows],dtype=torch.float64,device=self.device),counts,row_nodes=self.nodes[rows])
        pairs=[self.geometry(int(row)) for row in rows]
        return reference,[x[0] for x in pairs],[x[1] for x in pairs]

    def physical_batch(self,rows):
        return ([self.physical["dimensions"][i,self.pack["agent_mask"][i]] for i in rows],
                [self.physical["road_boundaries"][i,self.physical["road_boundary_mask"][i]] for i in rows])
