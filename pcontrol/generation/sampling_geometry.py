"""Counted sampling-time geometry; same validated PET and all FD checks."""
import numpy as np
import torch
from pcontrol.research.probe_natural_risk_directions import GeometryOnlyAdapter
from pcontrol.data.scene_pet_broadphase import scene_occupancy_pet as _scene_occupancy_pet


class SamplingGeometry(GeometryOnlyAdapter):
    def __init__(self, dimensions, anchors):
        super().__init__(dimensions, anchors)
        self.metric_calls = 0

    def _metric(self, *args, **kwargs):
        self.metric_calls += 1
        return _scene_occupancy_pet(*args, **kwargs)

    def score_future(self,candidate):
        future,_=self._candidate(candidate)
        metric=self._metric(future,np.ones(future.shape[:2],bool),self._dimensions,
            times=np.arange(175)*.04,ego_index=0)
        return dict(pet_seconds=metric['pet_value_seconds'],pet_raw_seconds=metric['observed_min_seconds'],
            metric=metric,CDF_or_rank_computed=False)

    def active_witness_pet(self, candidate):
        """Conservative local active-set PET derivative, with exact-oracle checks.

        Returns unsupported instead of inventing a gradient at zero, infinity,
        cap, competing actors, degenerate vertices or a failed local finite-
        difference test. This does not differentiate actor/witness selection,
        promise global control, or supply a CDF gradient. The observed t0 is a
        fixed boundary condition and has zero decision-variable gradient.
        ``value`` remains connected to the supplied candidate tensor; a caller
        may guide toward target_pet with it only while the local support holds.
        """
        if (not isinstance(candidate, torch.Tensor) or not candidate.requires_grad
                or candidate.dtype not in (torch.float32, torch.float64)):
            return dict(supported=False, reason='requires_float_candidate_with_grad', value=None, gradient=None)
        score = self.score_future(candidate)
        raw = score['pet_raw_seconds']
        unsupported = dict(supported=False, value=None, gradient=None, exact_score=score,
                           global_control_guaranteed=False)
        if not np.isfinite(raw) or not 0 < raw < 4.:
            return dict(unsupported, reason='zero_infinite_or_capped_PET')
        metric, witness = score['metric'], score['metric']['witness']
        finite = sorted(p['observed_min_seconds'] for p in metric['pair_results']
                        if np.isfinite(p['observed_min_seconds']))
        if len(finite) > 1 and finite[1]-finite[0] <= 1e-7:
            return dict(unsupported, reason='nonunique_critical_actor')
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
            return dict(unsupported,reason='oracle_witness_infeasible_under_reconstruction')
        active = np.flatnonzero(np.abs(residual) <= tolerance)
        if len(active) != 2:
            return dict(unsupported,reason='degenerate_or_unresolved_active_vertex',active_constraints=[names[j] for j in active])
        basis = M[active]
        if not np.isfinite(np.linalg.cond(basis)) or np.linalg.cond(basis) > 1e8:
            return dict(unsupported,reason='singular_or_ill_conditioned_active_basis')
        take = torch.tensor(active,device=matrix.device,dtype=torch.long)
        uv = torch.linalg.solve(matrix[take],rhs[take])
        value = torch.abs(uv[0]*((e1-e0)*.04)-uv[1]*((a1-a0)*.04)+de.new_tensor((e0-a0)*.04))
        replay_error = abs(float(value.detach().cpu())-raw)
        if replay_error > 1e-8 or not value.requires_grad:
            return dict(unsupported,reason='active_value_not_replayed_or_locally_constant',replay_error=replay_error)
        gradient = torch.autograd.grad(value,candidate,retain_graph=True,allow_unused=True)[0]
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            return dict(unsupported,reason='nonfinite_or_disconnected_active_gradient')
        coordinates = np.argwhere(np.abs(gradient.detach().cpu().numpy()) > 1e-9)
        if len(coordinates) == 0:
            return dict(unsupported,reason='flat_active_gradient')
        # Test the actual min-over-all-actors oracle, not just this vertex.
        # This detects common same-pair competing-time witnesses that would
        # otherwise produce a misleading nonzero local derivative.
        proposed,_ = self._candidate(candidate)
        checks = []
        for frame,container,channel in coordinates:
            if frame == 0 or channel >= 2:
                return dict(unsupported,reason='unexpected_anchor_or_velocity_gradient')
            local = int(np.flatnonzero(self._valid_indices == container)[0]) \
                if candidate.shape[1] == self._container_agents else int(container)
            step = 1e-6*max(1.,abs(float(proposed[frame,local,channel])))
            observations = []
            for direction in (-1.,1.):
                perturbed = proposed.copy();perturbed[frame,local,channel] += direction*step
                result = self._metric(perturbed,np.ones(perturbed.shape[:2],bool),self._dimensions,
                    times=np.arange(175)*.04,ego_index=self._ego_index)
                observations.append(result['observed_min_seconds'])
            if not np.isfinite(observations).all():
                return dict(unsupported,reason='witness_disappears_under_local_perturbation',
                            coordinate=[int(frame),int(container),int(channel)])
            derivative = (observations[1]-observations[0])/(2.*step)
            analytic = float(gradient[frame,container,channel].detach().cpu())
            if (not np.isfinite(derivative)
                    or abs(derivative-analytic) > 5e-4+.03*abs(analytic)):
                return dict(unsupported,reason='full_oracle_local_finite_difference_disagreement',
                            coordinate=[int(frame),int(container),int(channel)],
                            analytic_derivative=analytic,finite_difference_derivative=float(derivative))
            checks.append(dict(coordinate=[int(frame),int(container),int(channel)],
                               analytic=analytic,finite_difference=float(derivative),step=step))
        return dict(supported=True,reason='finite_positive_stable_active_set_locally_verified',
                    value=value,gradient=gradient,exact_score=score,
                    active_constraints=[names[j] for j in active],value_replay_error=replay_error,
                    finite_difference_checks=checks,anchor_gradient_fixed=True,
                    global_control_guaranteed=False,
                    validity='local_active_set_only_recompute_after_each_sampling_update')
