"""Detached mixed scene CDF + monotone warp; only query y can carry gradients.

No network, checkpoint, dataset, fitting or file access. At physical/probability
knots the selected right-hand piece is a one-sided convention, NOT smoothness.
Endpoint atoms have exact left/right values and zero query gradient.
"""
import torch
from torch import nn

from .scene_calibration import SceneCDFWarp, U_KNOTS


class FrozenTorchCDF(nn.Module):
    """Frozen [B,66] masses, [B] counts and one warp or prebuilt [B,9] nodes.

    ``cdf`` and ``rank`` accept paired y[B]/y[B,Q] (or shared y[1,Q]) and
    preserve those output shapes after batch broadcasting. References are
    copied/detached FP64 buffers; FP32/FP64 y retains its autograd connection.
    A single SceneCDFWarp is expanded using its global/count formula once.
    Cross-fold callers may instead provide independently built row_nodes.
    """
    VERSION = 'frozen_torch_mixed_scene_CDF_probability_warp_v1'

    def __init__(self, masses, counts, warp=None, base_knots=None, *, row_nodes=None):
        super().__init__()
        device = masses.device if isinstance(masses, torch.Tensor) else torch.device('cpu')

        def freeze(value):
            if isinstance(value, torch.Tensor):
                return value.detach().clone().to(device=device, dtype=torch.float64)
            return torch.tensor(value.copy() if hasattr(value, 'copy') else value,
                                dtype=torch.float64, device=device)

        mass, count = freeze(masses), freeze(counts)
        knots = (torch.linspace(0., 4., 65, dtype=torch.float64, device=device)
                 if base_knots is None else freeze(base_knots))
        u_knots = freeze(U_KNOTS)
        if (mass.ndim != 2 or mass.shape[0] < 1 or mass.shape[1] != 66
                or count.shape != mass.shape[:1] or knots.shape != (65,)
                or not bool(torch.isfinite(mass).all()) or bool((mass < 0).any())
                or not torch.allclose(mass.sum(-1), torch.ones_like(count), atol=2e-12, rtol=0.)
                or not bool(torch.isfinite(count).all()) or bool((count < 1).any())
                or bool((count != count.floor()).any()) or not bool(torch.isfinite(knots).all())
                or knots[0] != 0 or knots[-1] != 4 or not bool((knots[1:] > knots[:-1]).all())):
            raise ValueError('normalized [B,66] masses, integer counts[B], and increasing65 knots on[0,4] required')
        if row_nodes is not None:
            if warp is not None:
                raise ValueError('pass either one warp or prebuilt row_nodes, not both')
            nodes = freeze(row_nodes)
        elif warp is None:
            nodes = u_knots[None].expand(len(mass), -1).clone()
        else:
            if not isinstance(warp, SceneCDFWarp):
                raise TypeError('warp must be one SceneCDFWarp; prebuild row_nodes for multiple warps')
            source = freeze(warp.node_values)
            if warp.family == 'global':
                nodes = source[:1].expand(len(mass), -1).clone()
            else:
                blend = ((count - 5.) / 8.).clamp(0., 1.)[:, None]
                nodes = (1. - blend) * source[0] + blend * source[1]
        if (nodes.shape != (len(mass), 9) or not bool(torch.isfinite(nodes).all())
                or bool((nodes[:, 0] != 0).any()) or bool((nodes[:, -1] != 1).any())
                or bool((nodes[:, 1:] < nodes[:, :-1]).any())):
            raise ValueError('row_nodes must be finite monotone[B,9] with exact endpoints0/1')
        starts = mass[:, :1] + torch.cat((torch.zeros_like(mass[:, :1]),
                                         mass[:, 1:-2].cumsum(-1)), -1)
        # Preserve exact probability-coordinate provenance at representable
        # interior warp crossings. These frozen coordinates also flag kinks.
        ends = mass[:, :1] + mass[:, 1:-1].cumsum(-1)
        levels = u_knots[1:-1][None].expand(len(mass), -1).contiguous()
        index = torch.searchsorted(ends.contiguous(), levels, right=False).clamp(max=63)
        chosen_mass = mass[:, 1:-1].gather(1, index)
        crossings = knots[index] + ((levels - starts.gather(1, index))
                    / torch.where(chosen_mass > 0, chosen_mass, torch.ones_like(chosen_mass))) * (knots[1:] - knots[:-1])[index]
        crossing_valid = ((levels > mass[:, :1]) & (levels < 1. - mass[:, -1:])
                          & (chosen_mass > 0) & (crossings > 0) & (crossings < 4))
        for name, value in dict(masses=mass, counts=count, base_knots=knots,
                                u_knots=u_knots, row_nodes=nodes, bin_starts=starts,
                                warp_crossings=crossings, warp_crossing_valid=crossing_valid).items():
            self.register_buffer(name, value.detach().clone())

    @property
    def knots(self):
        return self.base_knots

    @property
    def cap_seconds(self):
        return 4.0

    def _query(self, y):
        if not isinstance(y, torch.Tensor) or y.dtype not in (torch.float32, torch.float64):
            raise TypeError('query y must be a float32/float64 tensor')
        if y.device != self.masses.device:
            raise ValueError('query and frozen reference must share device')
        if y.ndim not in (1, 2) or y.shape[0] not in (1, len(self.masses)) or y.numel() == 0:
            raise ValueError('query must be paired[B], paired[B,Q], or explicit shared[1,Q]')
        if y.ndim == 1 and y.shape[0] != len(self.masses):
            raise ValueError('one-dimensional query must pair exactly with B rows')
        if bool(torch.isnan(y).any()):
            raise ValueError('CDF query cannot contain NaN')
        query = y.to(torch.float64).expand((len(self.masses),) + y.shape[1:])
        return query, query.reshape(len(self.masses), -1)

    def _pieces(self, q):
        safe = q.clamp(0., 4.)
        index = torch.searchsorted(self.base_knots, safe.contiguous(), right=True).sub(1).clamp(0, 63)
        widths = self.base_knots[1:] - self.base_knots[:-1]
        base_density = self.masses[:, 1:-1].gather(1, index) / widths[index]
        probability = (self.bin_starts.gather(1, index)
                       + base_density * (safe - self.base_knots[index])).clamp(0., 1.)
        crossing_match = ((q[..., None] == self.warp_crossings[:, None])
                          & self.warp_crossing_valid[:, None])
        at_crossing = crossing_match.any(-1)
        exact_level = (crossing_match.to(q.dtype) * self.u_knots[None, None, 1:-1]).sum(-1)
        # Correct only the known coordinate's floating-point roundtrip; retain
        # the base derivative and select the right-hand warp piece there.
        probability = torch.where(at_crossing, probability + (exact_level - probability).detach(), probability)
        warp_index = torch.searchsorted(self.u_knots, probability.contiguous(), right=True).sub(1).clamp(0, 7)
        low = self.row_nodes.gather(1, warp_index)
        slope = ((self.row_nodes.gather(1, warp_index + 1) - low)
                 / (self.u_knots[warp_index + 1] - self.u_knots[warp_index]))
        calibrated = low + slope * (probability - self.u_knots[warp_index])
        return calibrated, base_density * slope, probability, at_crossing

    def _warp_constant(self, probability):
        index = torch.searchsorted(self.u_knots, probability.contiguous(), right=True).sub(1).clamp(0, 7)
        low = self.row_nodes.gather(1, index)
        high = self.row_nodes.gather(1, index + 1)
        return low + (high - low) * ((probability - self.u_knots[index])
                    / (self.u_knots[index + 1] - self.u_knots[index]))

    def cdf(self, y, side='right'):
        if side not in ('left', 'right'):
            raise ValueError('CDF side must be left or right')
        query, q = self._query(y)
        value, _, _, _ = self._pieces(q)
        zero = self._warp_constant(self.masses[:, :1])
        cap_left = self._warp_constant(1. - self.masses[:, -1:])
        value = torch.where(q == 0, zero if side == 'right' else torch.zeros_like(value), value)
        value = torch.where(q == 4, torch.ones_like(value) if side == 'right' else cap_left, value)
        value = torch.where(q < 0, torch.zeros_like(value), torch.where(q > 4, torch.ones_like(value), value))
        # Floating-point interpolation must not escape a probability bound;
        # this clips no observed y and introduces no epsilon or smoothing.
        return value.clamp(0., 1.).reshape(query.shape)

    def rank(self, y):
        query, _ = self._query(y)
        if not bool(torch.isfinite(query).all()) or bool(((query < 0) | (query > 4)).any()):
            raise ValueError('rank requires finite actual PET within[0,4]; no clipping')
        left, right = self.cdf(y, 'left'), self.cdf(y, 'right')
        return dict(p_low=1. - right, p_up=1. - left, p_mid=1. - .5 * (left + right))

    def density(self, y):
        """Right-piece derivative plus conservative validity/kink diagnostics.

        A zero/plateau density, endpoint, or physical/warp knot is invalid for
        a positive smooth local-density assumption. No smoothing is introduced.
        """
        query, q = self._query(y)
        _, slope, probability, at_crossing = self._pieces(q)
        interior = (q > 0) & (q < 4) & torch.isfinite(q)
        base_knot = (q[..., None] == self.base_knots).any(-1)
        warp_knot = at_crossing | (probability[..., None] == self.u_knots).any(-1)
        slope = torch.where(interior, slope, torch.zeros_like(slope))
        valid = interior & ~base_knot & ~warp_knot & torch.isfinite(slope) & (slope > 0)
        return {key: value.reshape(query.shape) for key, value in
                dict(density=slope, valid=valid, at_base_knot=base_knot,
                     at_warp_knot=warp_knot, interior=interior).items()}

    forward = cdf


TorchFrozenCDF = FrozenTorchCDF
