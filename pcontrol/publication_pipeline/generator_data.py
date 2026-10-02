"""Authenticated guarded-reference labels for the existing natural generator.

No legacy hard-coded policy or output-root mutation. Pure geometry-cache and
batch routines are reused; CDF-dependent ranks, conditions and tangents are new.
"""
import numpy as np
import torch

from pcontrol.time_attention_pipeline import common as c
from pcontrol.time_attention_pipeline.generator_data import join_rows,regenerate_tangent,TransformerFITTeacher
from pcontrol.reference.direct_p_crossfit import _load_pack
from pcontrol.reference.torch_frozen_cdf import FrozenTorchCDF
from pcontrol.reference.torch_frozen_inverse import PIECES_KEY,quantile_from_pieces
from pcontrol.generation.torch_atom_aware_target import midrank_target_from_pieces
from pcontrol.research import train_natural_scene_reference as io
from pcontrol.research import train_natural_diffusion as prior
from pcontrol.research import train_natural_direct_p as direct

PROTOCOL='guarded_reference_generator_adaptation_v1'


def same_binding(a,b):
    return io.verify_binding(a).resolve()==io.verify_binding(b).resolve() and a['sha256']==b['sha256']


def load_inputs(pb):
    p=c.json_file(pb)
    if (p['protocol']!=PROTOCOL or not p['only_natural_observations'] or p['protected_data_access']
            or p['multi_training_seed_campaign']):raise ValueError('wrong generator adaptation scope')
    lm=c.json_file(p['labels_manifest']);audit=c.json_file(p['reference_chain_audit'])
    if (lm['status']!='complete' or audit['status']!='pass'
            or not same_binding(lm['reference_manifest'],p['reference_manifest'])
            or not same_binding(audit['reference_manifest'],p['reference_manifest'])
            or not same_binding(audit['labels_manifest'],p['labels_manifest'])):
        raise ValueError('audited current reference/OOF chain required')
    c.verify_sources(lm['code_sha256'])
    data=c.json_file(p['generator_data']);dp=c.json_file(data['policy'])
    prior.validate_prepared_metadata(data,dp,data['policy'])
    packs={};labels={};contexts={};pvalues={};joins={}
    for role,count in (('FIT',9913),('STOP',538)):
        pack=prior.load_pack(data['packs'][role],role)
        lab=join_rows(pack,c.arrays(lm['roles'][role]['labels']))
        ctx=join_rows(pack,c.arrays(lm['roles'][role]['context']))
        values,evidence=direct.join_percentile_labels(pack,lab,role=role)
        if (len(values)!=count or not np.array_equal(ctx['pet_seconds'],lab['pet_seconds'])
                or not np.array_equal(ctx['num_agents'],pack['agent_mask'].sum(1))
                or not np.array_equal(ctx['fold'],lab['fold'])):raise ValueError('label/context/roster mismatch')
        reference=FrozenTorchCDF(ctx['joint_masses'],ctx['num_agents'],row_nodes=ctx['row_nodes'])
        replay=reference.rank(torch.tensor(lab['pet_seconds']))['p_mid'].numpy()
        error=float(abs(replay-values).max())
        if error>1e-12:raise ValueError('natural ranks fail own-CDF replay')
        pieces=torch.tensor(ctx[PIECES_KEY]);risk=torch.tensor(values)
        q=quantile_from_pieces(pieces,1.-risk)
        target=midrank_target_from_pieces(pieces,risk)
        canonical_error=float(abs(reference.rank(q)['p_mid'].numpy()-values).max())
        target_error=float(abs(reference.rank(target)['p_mid'].numpy()-values).max())
        if target_error>1e-10:raise ValueError('natural attainable rank fails atom-aware target replay')
        # Canonical plateaus need not return the original physical observation.
        # The canonical rank replay is recorded, not an assumption in our loader.
        evidence.update(own_CDF_replay_max_error=error,natural_label_canonical_rank_error=canonical_error,
                        natural_label_atom_target_rank_error=target_error)
        packs[role],labels[role],contexts[role],pvalues[role],joins[role]=pack,lab,ctx,values,evidence
    if set(packs['FIT']['recording_id'])&set(packs['STOP']['recording_id']):raise ValueError('recording leakage')
    physical=join_rows(packs['FIT'],_load_pack(lm['physical_packs']['FIT'],'FIT'));mask=packs['FIT']['agent_mask']
    if (not np.array_equal(physical['target'],labels['FIT']['pet_seconds'])
            or not np.array_equal(physical['agent_mask'],mask)
            or not np.array_equal(physical['history'][:,-1][mask],packs['FIT']['anchors'][mask])):
        raise ValueError('natural physical identity changed')
    return dict(policy=p,data=data,packs=packs,labels=labels,contexts=contexts,pvalues=pvalues,joins=joins,
        physical=physical,labels_manifest=lm,labels_binding=p['labels_manifest'])
