"""Training-only current-generated-PET value objective for variable-N batches.

This module does not load data, optimize weights, or change a sampling path.
One geometry adapter and decoder per real scene are supplied by a FIT-only
caller. It never substitutes cached natural tangents for current geometry.
"""
from collections import Counter

import torch

from .terminal_percentile_loss import terminal_percentile_value_loss
from .pet_target_value_loss import pet_target_value_loss


def terminal_rank_and_pet_losses(coefficients, requested_p, reference, decoders, geometries,
                        agent_mask, condition_present, *, target_pet, beta=.05, minimum_density=1e-6, pet_beta=.05):
    batch = coefficients.shape[0]
    if (coefficients.ndim != 4 or agent_mask.shape != coefficients.shape[:2]
            or agent_mask.dtype != torch.bool or agent_mask.device != coefficients.device
            or len(decoders) != batch or len(geometries) != batch
            or requested_p.shape != (batch,) or condition_present.shape != (batch,)
            or condition_present.dtype != torch.bool or condition_present.device != coefficients.device
            or not coefficients.requires_grad):
        raise ValueError('one decoder/geometry per differentiable masked scene and explicit p presence required')
    values, support, reasons = [], [], []
    for i, (decode, geometry) in enumerate(zip(decoders, geometries)):
        future = decode(coefficients[i, agent_mask[i]])
        active = geometry.active_witness_pet(future)
        score = active.get('exact_score')
        if score is None:
            score = geometry.score_future(future)
        exact = float(score['pet_seconds'])
        if not 0 <= exact <= 4:
            raise ValueError('finite exact capped current PET required')
        if active['supported']:
            value = active['value']
            if (value.ndim != 0 or not value.requires_grad or not bool(torch.isfinite(value))
                    or abs(float(value.detach()) - exact) > 1e-8):
                raise ValueError('supported current geometry must replay exact PET with its graph')
        else:
            value = coefficients.new_tensor(exact, dtype=torch.float64)
        values.append(value); support.append(bool(active['supported'])); reasons.append(active['reason'])
    pet = torch.stack(values)
    result = terminal_percentile_value_loss(pet, requested_p, reference,
        supported_geometry=torch.tensor(support, dtype=torch.bool, device=coefficients.device),
        condition_present=condition_present, beta=beta, minimum_density=minimum_density,
        graph_anchor=coefficients)
    result['geometry_reasons'] = dict(Counter(reasons))
    result['geometry_supported_scenes'] = sum(support)
    result['condition_present_scenes'] = int(condition_present.sum())
    result['exact_current_PET'] = pet.detach()
    result['PET_target_value'] = pet_target_value_loss(pet,target_pet,
        torch.tensor(support,dtype=torch.bool,device=coefficients.device),condition_present,
        beta=pet_beta,graph_anchor=coefficients)
    result['current_geometry_per_request'] = reasons
    return result

