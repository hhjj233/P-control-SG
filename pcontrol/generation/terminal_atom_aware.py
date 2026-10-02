"""Current-trajectory atom-aware loss; no cached-observation geometry gradient.

All requests retain their original midpoint-rank error and Fine denominator.
Only the training numeric residual subtracts the frozen scalar error infimum.
Physical targets/encounter support use the selected attainable scalar target;
the Fine surrogate remains limited to exactly target-eligible requests.
"""
from collections import Counter
import torch
from .terminal_percentile_loss import terminal_percentile_value_loss
from .pet_target_value_loss import pet_target_value_loss
from .fine_band_loss import fine_band_loss
from .excess_rank_loss import excess_midrank_value_loss


def terminal_atom_losses(coefficients,requested_p,reference,decoders,geometries,agent_mask,condition_present,
                         *,target_pet,scalar_error_infimum,exact_target_eligible,
                         beta=.05,minimum_density=1e-6,pet_beta=.05,fine_temperature=.01):
    batch=coefficients.shape[0]
    if (coefficients.ndim!=4 or agent_mask.shape!=coefficients.shape[:2] or agent_mask.dtype!=torch.bool
            or agent_mask.device!=coefficients.device or len(decoders)!=batch or len(geometries)!=batch
            or requested_p.shape!=(batch,) or condition_present.shape!=(batch,)
            or condition_present.dtype!=torch.bool or condition_present.device!=coefficients.device
            or exact_target_eligible.shape!=(batch,) or exact_target_eligible.dtype!=torch.bool
            or exact_target_eligible.device!=coefficients.device or not coefficients.requires_grad):
        raise ValueError('current differentiable geometry and explicit aligned request masks required')
    values=[];support=[];reasons=[]
    for i,(decode,geometry) in enumerate(zip(decoders,geometries)):
        future=decode(coefficients[i,agent_mask[i]])
        active=geometry.active_witness_pet(future);score=active.get('exact_score')
        if score is None:score=geometry.score_future(future)
        exact=float(score['pet_seconds'])
        if not 0<=exact<=4:raise ValueError('finite current capped PET required')
        if active['supported']:
            value=active['value']
            if (value.ndim!=0 or not value.requires_grad or not bool(torch.isfinite(value))
                    or abs(float(value.detach())-exact)>1e-8):raise ValueError('current PET graph does not replay exact geometry')
        else:value=coefficients.new_tensor(exact,dtype=torch.float64)
        values.append(value);support.append(bool(active['supported']));reasons.append(active['reason'])
    pet=torch.stack(values);geometry_support=torch.tensor(support,dtype=torch.bool,device=coefficients.device)
    result=terminal_percentile_value_loss(pet,requested_p,reference,supported_geometry=geometry_support,
        condition_present=condition_present,beta=beta,minimum_density=minimum_density,graph_anchor=coefficients)
    rank=reference.rank(pet)['p_mid']
    excess=excess_midrank_value_loss(rank,requested_p,scalar_error_infimum,result['support_mask'],beta=beta,graph_anchor=coefficients)
    result['original_numeric_loss']=result['loss']
    result['loss']=excess['loss'];result['per_scene_value_loss']=excess['per_scene_value_loss']
    result.update(geometry_reasons=dict(Counter(reasons)),geometry_supported_scenes=sum(support),
        condition_present_scenes=int(condition_present.sum()),exact_current_PET=pet.detach(),current_geometry_per_request=reasons,
        scalar_error_infimum=scalar_error_infimum.detach(),all_request_excess_error=excess['all_request_excess_error'],
        PET_target_value=pet_target_value_loss(pet,target_pet,geometry_support,condition_present,beta=pet_beta,graph_anchor=coefficients),
        Fine_band=fine_band_loss(rank,requested_p,result['support_mask']&exact_target_eligible,
            temperature=fine_temperature,graph_anchor=coefficients),
        numeric_objective='SmoothL1(max(abs(P_hat-P)-scalar_infimum,0))',primary_point_metrics_unchanged=True)
    return result
