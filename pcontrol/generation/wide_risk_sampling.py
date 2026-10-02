"""Broader FIT-only history coverage and same-H common-noise P requests."""
import numpy as np
import torch
from .slow_history_sampling import SlowHistoryCycle


class WideRiskHistoryCycle(SlowHistoryCycle):
    """Twelve natural random rows + two creep + two slow, all distinct."""
    def select(self,base_rows):
        rows=list(map(int,base_rows[:12]))
        if len(rows)!=12 or len(set(rows))!=12:raise ValueError('12 distinct base FIT rows required')
        for group in ('creep','slow'):
            for _ in range(2):rows.append(self._pick(group,set(rows)))
        return np.asarray(rows,dtype=np.int64)


def coupled_request_noise(shape,histories,slots,*,generator):
    """One fresh z per H/update, shared only by its different requested P's."""
    if len(shape)!=4 or shape[0]!=histories*slots or histories<1 or slots!=3:
        raise ValueError('B=histories*3, N,K,2 coefficient noise shape required')
    return torch.randn((histories,)+tuple(shape[1:]),generator=generator).repeat_interleave(slots,0)
