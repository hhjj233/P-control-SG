"""Local PET direction proposals, verified later by coefficient-space probes."""
import numpy as np
import torch
from .sampling_geometry import SamplingGeometry


class DirectionalSamplingGeometry(SamplingGeometry):
    def local_candidate(self, candidate):
        """Propose an active-vertex derivative, WITHOUT ambient FD validation.

        Availability means only that a finite local vertex derivative could
        be reconstructed. The caller must measure the full-scene derivative
        along its actual coefficient direction before taking a risk step.
        Zero/cap, tied actors and degenerate vertices remain unavailable.
        This is not a full PET-gradient or global-control guarantee.
        """
        if (not isinstance(candidate, torch.Tensor) or not candidate.requires_grad
                or candidate.dtype not in (torch.float32, torch.float64)):
            return dict(available=False, reason='requires_float_candidate_with_grad', value=None, gradient=None)
        score = self.score_future(candidate)
        raw = score['pet_raw_seconds']
        unavailable = dict(available=False, value=None, gradient=None, exact_score=score,
                           global_control_guaranteed=False)
        if not np.isfinite(raw) or not 0 < raw < 4.:
            return dict(unavailable, reason='zero_infinite_or_capped_PET')
        metric, witness = score['metric'], score['metric']['witness']
        finite = sorted(p['observed_min_seconds'] for p in metric['pair_results']
                        if np.isfinite(p['observed_min_seconds']))
        if len(finite) > 1 and finite[1]-finite[0] <= 1e-7:
            return dict(unavailable, reason='nonunique_critical_actor')
        other = int(witness['other_index'])
        e0,e1 = witness['ego_segment_frames']; a0,a1 = witness['other_segment_frames']
        if candidate.shape[1] == self._container_agents:
            ei, ai = int(self._valid_indices[self._ego_index]), int(self._valid_indices[other])
        else:
            ei, ai = self._ego_index, other
        def point(frame, actor):
            value = candidate[frame, actor, :2].to(torch.float64)
            return value.detach() if frame == 0 else value
        E0,E1,A0,A1 = point(e0,ei),point(e1,ei),point(a0,ai),point(a1,ai)
        de,da,delta = E1-E0,A1-A0,E0-A0
        zero,one = de.new_tensor(0.),de.new_tensor(1.)
        rows = [torch.stack((-one,zero)),torch.stack((one,zero)),
                torch.stack((zero,-one)),torch.stack((zero,one))]
        bounds = [zero,one,zero,one]
        names = ['u_lower','u_upper','v_lower','v_upper']
        half = .5*(self._dimensions[self._ego_index]+self._dimensions[other])
        for axis,name in enumerate(('x','y')):
            normal = torch.stack((de[axis],-da[axis]))
            rows.extend((normal,-normal))
            bounds.extend((de.new_tensor(half[axis])-delta[axis],de.new_tensor(half[axis])+delta[axis]))
            names.extend((name+'_upper',name+'_lower'))
        matrix,rhs = torch.stack(rows),torch.stack(bounds)
        M,b = matrix.detach().cpu().numpy(),rhs.detach().cpu().numpy()
        uv0 = np.asarray(witness['segment_coordinates_uv'],dtype=np.float64)
        residual = b-M@uv0
        tolerance = 1e-9*np.maximum(1.,np.maximum(np.abs(b),np.max(np.abs(M),axis=1)))
        if np.any(residual < -tolerance):
            return dict(unavailable,reason='oracle_witness_infeasible_under_reconstruction')
        active = np.flatnonzero(np.abs(residual) <= tolerance)
        if len(active) != 2:
            return dict(unavailable,reason='degenerate_or_unresolved_active_vertex',active_constraints=[names[j] for j in active])
        basis = M[active]
        if not np.isfinite(np.linalg.cond(basis)) or np.linalg.cond(basis) > 1e8:
            return dict(unavailable,reason='singular_or_ill_conditioned_active_basis')
        take = torch.tensor(active,device=matrix.device,dtype=torch.long)
        uv = torch.linalg.solve(matrix[take],rhs[take])
        value = torch.abs(uv[0]*((e1-e0)*.04)-uv[1]*((a1-a0)*.04)+de.new_tensor((e0-a0)*.04))
        replay_error = abs(float(value.detach().cpu())-raw)
        if replay_error > 1e-8 or not value.requires_grad:
            return dict(unavailable,reason='active_value_not_replayed_or_locally_constant',replay_error=replay_error)
        gradient = torch.autograd.grad(value,candidate,retain_graph=True,allow_unused=True)[0]
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            return dict(unavailable,reason='nonfinite_or_disconnected_active_gradient')
        coordinates = np.argwhere(np.abs(gradient.detach().cpu().numpy()) > 1e-9)
        if len(coordinates) == 0:
            return dict(unavailable,reason='flat_active_gradient')
        return dict(available=True,value=value,gradient=gradient,exact_score=score,reason='local_candidate_not_ambient_verified')
