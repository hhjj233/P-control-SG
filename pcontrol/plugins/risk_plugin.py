"""Backend-independent frozen natural-scene risk-reference algorithm plugin.

This is a Python research interface, not an application plugin. A backend
conditions once on physical H13/static geometry, queries the frozen CDF/target
PET, and scores proposed full futures through the unchanged all-actor oracle.
No candidate future or actor identity is sent to the reference estimator.
Estimated ranks are NOT known true percentiles or a safety certificate.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from pcontrol.reference.scene_calibration import SceneCDFWarp
from pcontrol.data.scene_pet import scene_occupancy_pet, PROTOCOL as PET_VERSION


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'frozen_natural_ego_scene_risk_plugin_v1'
REFINEMENT_SHA = '2c14d29a2ea252bc46e6e858f68872d6c7a6bb2d1581571f0e233ff3bebdbdcf'
CALIBRATION_SHA = 'f89ab7023eaab021fb1fb69b7df212dbc72448fe98a3c8c002d2c9cc5ec82bed'
CHECKPOINT_SHA = 'c214085f32754612d083e508dc17603c31b178640b16343ae1fc2ebd36fc8c6e'
NORMALIZER_SHA = 'da2cba7987c73e0d642fe098b392c1f46f4627b28baad765eced6135d3f46055'
ANCHOR_ATOL = 1e-5


def _resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT/path).resolve()


def _sha(path):
    digest = hashlib.sha256()
    with _resolve(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _bound(binding):
    if set(binding) != {'path','sha256'} or _sha(binding['path']) != binding['sha256']:
        raise ValueError('risk plugin artifact hash mismatch')
    return _resolve(binding['path'])


def _json(binding):
    return json.loads(_bound(binding).read_text())


def _readonly(value, dtype=np.float64):
    value = np.array(value, dtype=dtype, copy=True)
    value.setflags(write=False)
    return value


def _unbatch(value):
    value = np.asarray(value)[0]
    return float(value) if value.ndim == 0 else _readonly(value)


@runtime_checkable
class RiskPlugin(Protocol):
    def describe(self): ...
    def condition(self, history, dimensions, road_boundaries, ego_mask, agent_mask): ...


@dataclass(frozen=True)
class RiskReference:
    """One immutable history-conditioned distribution; future-independent queries."""
    _masses: np.ndarray
    _warp: SceneCDFWarp
    _knots: np.ndarray
    _dimensions: np.ndarray
    _t0: np.ndarray
    _valid_indices: np.ndarray
    _container_agents: int
    _ego_index: int
    _metadata: dict

    @property
    def num_agents(self):
        return len(self._dimensions)

    def describe(self):
        return dict(copy.deepcopy(self._metadata), num_agents=self.num_agents,
                    candidate_t0_absolute_tolerance=ANCHOR_ATOL,
                    original_valid_container_indices=self._valid_indices.tolist())

    def cdf(self, y, side='right'):
        query = np.asarray(y, dtype=np.float64)
        return _unbatch(self._warp.cdf(self._masses, [self.num_agents], query[None],
                                       side=side, base_knots=self._knots))

    def quantile(self, u):
        query = np.asarray(u, dtype=np.float64)
        return _unbatch(self._warp.quantile(self._masses, [self.num_agents], query[None],
                                            base_knots=self._knots))

    def target_pet(self, p):
        p = np.asarray(p, dtype=np.float64)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('requested adversity p must lie in [0,1]')
        return self.quantile(1.-p)

    def rank(self, y):
        query = np.asarray(y, dtype=np.float64)
        answer = self._warp.rank(self._masses, [self.num_agents], query[None], base_knots=self._knots)
        return {key: _unbatch(value) for key, value in answer.items()}

    def target_spec(self, p):
        target = self.target_pet(p)
        rank = self.rank(target)
        width = np.asarray(rank['p_up'])-np.asarray(rank['p_low'])
        return dict(requested_p=np.asarray(p).tolist(), target_pet_seconds=np.asarray(target).tolist(),
                    estimated_rank_interval=[np.asarray(rank['p_low']).tolist(), np.asarray(rank['p_up']).tolist()],
                    estimated_midrank=np.asarray(rank['p_mid']).tolist(),
                    atom_rank_interval_width=width.tolist(),
                    exact_point_rank_identified=(width == 0).tolist(),
                    half_atom_rank_width=(width/2).tolist(),
                    requested_p_midpoint_distance=np.abs(np.asarray(p)-np.asarray(rank['p_mid'])).tolist(),
                    caveat='atom-interior p values share a PET target; no artificial within-atom label is assigned')

    def _candidate(self, candidate):
        if isinstance(candidate, torch.Tensor):
            candidate = candidate.detach().cpu().numpy()
        values = np.asarray(candidate, dtype=np.float64)
        if values.shape == (175, self._container_agents, 4):
            values = values[:, self._valid_indices]
        elif values.shape != (175, self.num_agents, 4):
            raise ValueError('candidate must contain all175 native future states of the conditioned fixed roster')
        if not np.isfinite(values).all():
            raise ValueError('candidate future must be complete and finite; missing paths are not filled')
        error = float(np.max(np.abs(values[0]-self._t0)))
        if error > ANCHOR_ATOL:
            raise ValueError('candidate t0 differs from the conditioned observed history')
        return values, error

    def score_future(self, candidate):
        """Exact all-actor functional on a proposal, not a new natural-data label."""
        future, anchor_error = self._candidate(candidate)
        metric = scene_occupancy_pet(future, np.ones(future.shape[:2], bool), self._dimensions,
            times=np.arange(175)*.04, ego_index=self._ego_index, sample_period=.04,
            window=(0.,6.96), cap_seconds=4.)
        target = metric['pet_value_seconds']
        rank = self.rank(target)
        critical = metric['critical_other_index']
        return dict(pet_seconds=float(target), pet_raw_seconds=metric['observed_min_seconds'],
                    pet_raw_is_positive_infinity=bool(np.isposinf(metric['observed_min_seconds'])),
                    estimated_rank=rank, metric=metric,
                    critical_container_index=None if critical is None else int(self._valid_indices[critical]),
                    candidate_anchor_max_absolute_error=anchor_error,
                    candidate_origin='external_or_generated_proposal_not_asserted_natural_observation',
                    exact_functional_applied_to_supplied_candidate=True,
                    natural_ground_truth_percentile_known=False, plausibility_assessed=False,
                    online_safety_certified=False, estimator_rerun_on_future=False)

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
                result = scene_occupancy_pet(perturbed,np.ones(perturbed.shape[:2],bool),self._dimensions,
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


class FrozenRiskPlugin:
    """Read-only M2+count-warp wrapper; official construction is hash-bound.

    The constructor accepts in-memory components for software tests/integration,
    but only from_refinement_binding sets production_bound=True. No dataset,
    training package, source CSV, CAL or AUDIT prediction archive is decoded.
    """
    def __init__(self, model, normalizer, warp, *, provenance=None):
        if not isinstance(warp, SceneCDFWarp):
            raise TypeError('a frozen SceneCDFWarp is required')
        self._model = copy.deepcopy(model).eval()
        self._model.requires_grad_(False)
        self._normalizer = copy.deepcopy(normalizer)
        hscale = np.asarray(normalizer['history_scale'], dtype=np.float64)
        dscale = np.asarray(normalizer['dimension_scale'], dtype=np.float64)
        if (hscale.shape != (4,) or dscale.shape != (2,) or not np.isfinite(hscale).all()
                or not np.isfinite(dscale).all() or np.any(hscale < 1) or np.any(dscale < 1)
                or normalizer.get('history_centering') is not False
                or normalizer.get('roles_used') != ['FIT'] or normalizer.get('targets_used') is not False
                or normalizer.get('future_or_CAL_or_AUDIT_used') is not False
                or normalizer.get('road_scale_source') != 'history_scale_y'):
            raise ValueError('risk plugin requires the frozen history/static FIT-only scale contract')
        self._hscale, self._dscale, self._warp = _readonly(hscale), _readonly(dscale), warp
        self._metadata = dict(protocol=PROTOCOL, production_bound=False,
            provenance=copy.deepcopy(provenance or {}), estimator='M2_no_focal_ego_scene',
            target_estimand='P(Y_scene_6.96s <= y | H, E_full=1)',
            E_full='all_history_selected_agents_all175_native_future_frames_observed',
            metric_version=PET_VERSION, CDF_warp=warp.as_dict(),
            calibration_status='PIT_not_fully_calibrated_or_uniformly_improved',
            evaluation_scope='reused_internal_development_AUDIT_not_final_blind_test',
            encoder_inputs='history_dimensions_road_boundaries_ego_mask_agent_mask_padding_mask_only',
            future_or_actor_identity_estimator_inputs=False, model_training=False,
            true_conditional_percentile_claimed=False,
            generated_future_is_natural_observation=False,
            atom_rank_warning='zero/cap atoms identify a rank interval, not an exact within-atom p')

    @classmethod
    def from_refinement_binding(cls, binding, *, device='cpu'):
        """Authenticate the selected result before deserializing its known weights."""
        if binding.get('sha256') != REFINEMENT_SHA:
            raise ValueError('only the explicitly selected frozen refinement result is accepted')
        result = _json(binding)
        if (result.get('status') != 'complete' or result.get('protocol') != 'natural_M2_CAL_record_CV_refinement_v1'
                or result.get('selected_base_name') != 'highN'
                or result.get('selected_calibration_name') != 'count_ridge_0.1'
                or result['calibration_model']['sha256'] != CALIBRATION_SHA):
            raise ValueError('refinement result does not bind the selected M2/count-warp pair')
        prepared = _json(result['parent_prepared'])
        base = _json(result['base_selection'])
        calibration = _json(result['calibration_selection'])
        if (base.get('status') != 'base_frozen' or base.get('selected_name') != 'highN'
                or base.get('base_selected_before_CAL_decode') is not True
                or base['selected'] != base['candidates']['highN']
                or base['selected']['checkpoint']['sha256'] != CHECKPOINT_SHA
                or calibration.get('status') != 'calibration_frozen'
                or calibration['base_selection'] != result['base_selection']
                or calibration['calibration_model'] != result['calibration_model']
                or calibration.get('selected_name') != result['selected_calibration_name']
                or prepared['normalizer']['sha256'] != NORMALIZER_SHA
                or prepared['data'] != result['source_data']):
            raise ValueError('selected weights, calibration or normalizer lineage changed')
        code = dict(prepared['code_sha256'])
        code.update(result['code_sha256'])
        for path, checksum in code.items():
            _bound(dict(path=path, sha256=checksum))
        policy = _json(prepared['policy'])
        normalizer = _json(prepared['normalizer'])
        warp = SceneCDFWarp.from_dict(_json(result['calibration_model']))
        if warp.family != 'count':
            raise ValueError('selected risk plugin requires the frozen count warp')
        # Torch1.12 lacks weights_only. This exact internally produced checkpoint
        # is authenticated by the hard-coded approved SHA BEFORE pickle loading.
        checkpoint = torch.load(_bound(base['selected']['checkpoint']), map_location='cpu')
        if (checkpoint.get('protocol') != 'natural_M2_same_seed_continuation_v1'
                or checkpoint.get('recipe') != 'highN' or checkpoint['normalizer'] != prepared['normalizer']
                or checkpoint['data'] != prepared['data'] or checkpoint['policy'] != result['policy']
                or checkpoint['code_sha256']['parent_training'] != prepared['code_sha256']
                or checkpoint['code_sha256']['refinement'] != result['code_sha256']['pcontrol/research/refine_natural_scene_M2.py']):
            raise ValueError('checkpoint is not the frozen same-source highN M2')
        from pcontrol.reference.scene_models import SceneCDFReference
        m = policy['model']
        # Fresh temporary initialization is overwritten, and must not consume
        # the diffusion backend's externally controlled sampling RNG stream.
        with torch.random.fork_rng(devices=[]):
            model = SceneCDFReference('M2', torch.linspace(0., m['cap_seconds'], m['bins']+1, dtype=torch.float64),
                                      zero_atom_enabled=m['zero_atom_enabled'], hidden_dim=m['hidden_dim'], heads=m['heads'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.to(torch.device(device))  # No automatic GPU choice.
        provenance = dict(refinement_result=copy.deepcopy(binding), checkpoint=base['selected']['checkpoint'],
            normalizer=prepared['normalizer'], calibration_model=result['calibration_model'],
            base_selection=result['base_selection'], calibration_selection=result['calibration_selection'],
            reference_training_dataset=prepared['data'],
            frozen_code_sha256=code, plugin_code_sha256=_sha(__file__),
            checkpoint_and_normalizer_loaded_readonly=True, datasets_or_prediction_archives_decoded=False)
        plugin = cls(model, normalizer, warp, provenance=provenance)
        plugin._metadata['production_bound'] = True
        return plugin

    def describe(self):
        return copy.deepcopy(self._metadata)

    def condition(self, history, dimensions, road_boundaries, ego_mask, agent_mask):
        """Condition on ONE raw physical scene, not denoiser-normalized features.

        history[13,N,4], dimensions[N,2], road_boundaries[R] contain centre
        positions/velocities and metres in the frozen canonical fixed-road frame.
        Bool masks[N] identify padding and exactly one real ego. IDs, semantics,
        future labels and risk targets are not accepted as estimator inputs.
        """
        h, dims, road = (np.asarray(v, dtype=np.float64) for v in (history, dimensions, road_boundaries))
        ego, valid = np.asarray(ego_mask), np.asarray(agent_mask)
        if (h.ndim != 3 or h.shape[0] != 13 or h.shape[2] != 4 or h.shape[1] < 3
                or dims.shape != (h.shape[1], 2) or road.ndim != 1 or len(road) < 2
                or ego.shape != (h.shape[1],) or valid.shape != ego.shape
                or ego.dtype != np.bool_ or valid.dtype != np.bool_ or ego.sum() != 1
                or np.any(ego & ~valid) or valid.sum() < 3
                or not np.isfinite(h[:, valid]).all() or not np.isfinite(dims[valid]).all()
                or np.any(dims[valid] <= 0) or not np.isfinite(road).all() or np.any(np.diff(road) <= 0)):
            raise ValueError('valid physical complete H13/static context, >=3 real actors and exactly one ego required')
        clean_h = np.where(valid[None, :, None], h, 0.)
        clean_dims = np.where(valid[:, None], dims, 0.)
        features = dict(history=(clean_h[None]/self._hscale).astype(np.float32),
            dimensions=(clean_dims[None]/self._dscale).astype(np.float32),
            road_boundaries=(road[None]/self._hscale[1]).astype(np.float32),
            road_boundary_mask=np.ones((1,len(road)), bool), ego_mask=ego[None], agent_mask=valid[None])
        device = next(self._model.parameters()).device
        tensors = {key: torch.from_numpy(np.array(value, copy=True)).to(device) for key, value in features.items()}
        self._model.eval()
        with torch.inference_mode():
            params = self._model(tensors)
        indices = np.flatnonzero(valid)
        ego_position = int(np.flatnonzero(ego[valid])[0])
        return RiskReference(_readonly(params.joint_masses.cpu().numpy()), self._warp,
            _readonly(params.knots.cpu().numpy()), _readonly(dims[valid]), _readonly(h[-1,valid]),
            _readonly(indices, np.int64), h.shape[1], ego_position, self.describe())
