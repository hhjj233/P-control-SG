"""Public history-only natural diffusion prior, independent of any risk plugin.

The approved trained prior is loaded by immutable result/checkpoint bindings.
Sampling requires physical observed H13/static inputs and an explicit seed or
float32 initial z. A generic callback factory may be supplied externally; this
module neither imports nor loads a risk estimator, percentile target or guide.
No training/evaluation dataset archive is decoded by this interface.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .diffusion import JointHistoryDenoiser, CosineDiffusionSchedule, ddim_sample
from .trajectory_basis import TrajectoryBasis, basis_matrices


ROOT = Path(__file__).resolve().parents[2]
RESULT_SHA = 'f5a2f45d3e998e8a129ad9dba7eff86dda7afb88badf56f3a9a00a92c481c1b1'
CHECKPOINT_SHA = '2fc90eed85aa1dc73d0285890347f1bac89c662c79a14290ad054537070e6fc6'
POLICY_SHA = '5477744cd4a71eef90d0fd55882185cda50256e74f0eeb8703f55d4ed31d4997'
DATA_SHA = 'b10d412b54cf6a94c82639362515b4fff62305d8e448c5c6dbcd3eaa00fad408'
TRAIN_PROTOCOL = 'natural_history_only_diffusion_prior_pilot_v1'
PROTOCOL = 'public_frozen_natural_history_diffusion_prior_v1'


def _path(value):
    p = Path(value)
    return (p if p.is_absolute() else ROOT / p).resolve()


def _sha(path):
    digest = hashlib.sha256()
    with _path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _bound(binding):
    if (not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}
            or _sha(binding['path']) != binding['sha256']):
        raise ValueError('prior artifact binding mismatch')
    return _path(binding['path'])


def _json(binding):
    return json.loads(_bound(binding).read_text())


class _PhysicalDecoder:
    """Thin differentiable analytic basis decoder; no guidance/risk dependency."""
    def __init__(self, basis, mean, scale, anchors, device):
        self.basis = basis
        bp, bv, _ = basis_matrices(basis.times, basis.modes, basis.horizon)
        self.bp = torch.tensor(bp, dtype=torch.float64, device=device)
        self.bv = torch.tensor(bv, dtype=torch.float64, device=device)
        self.times = torch.tensor(basis.times, dtype=torch.float64, device=device)
        self.mean = torch.tensor(mean, dtype=torch.float64, device=device)
        self.scale = torch.tensor(scale, dtype=torch.float64, device=device)
        self.anchors = torch.tensor(anchors, dtype=torch.float64, device=device)

    def __call__(self, coefficients):
        if coefficients.shape != (len(self.anchors), self.basis.modes, 2):
            raise ValueError('decoder requires unpadded [N,K,2] normalized coefficients')
        c = coefficients.to(torch.float64) * self.scale + self.mean
        xy = (self.anchors[None, :, :2] + self.times[:, None, None] * self.anchors[None, :, 2:]
              + torch.einsum('tk,nkd->tnd', self.bp, c))
        velocity = self.anchors[None, :, 2:] + torch.einsum('tk,nkd->tnd', self.bv, c)
        return torch.cat((xy, velocity), -1)


class NaturalDiffusionPrior:
    """One frozen model, explicit K=1 DDIM path and external optional callback.

    Production construction is exclusively from_result_binding. The component
    constructor is available for software tests and reports production_bound
    false. All vehicles are actual unpadded actors; the unique ego may appear
    at any index. Changing a scene's actor order requires the same permutation
    of the supplied z to preserve a paired path.
    """
    def __init__(self, model, schedule, basis, coefficient_normalizer, history_normalizer,
                 *, prediction_type='v', sampling_steps=50, provenance=None):
        if (not isinstance(model, JointHistoryDenoiser) or not isinstance(schedule, CosineDiffusionSchedule)
                or not isinstance(basis, TrajectoryBasis) or model.coefficient_dim != 2 * basis.modes
                or prediction_type not in ('epsilon', 'v') or type(sampling_steps) is not int
                or not 1 <= sampling_steps <= schedule.steps):
            raise ValueError('compatible standard denoiser, schedule, basis and explicit parameterization required')
        cn, hn = copy.deepcopy(coefficient_normalizer), copy.deepcopy(history_normalizer)
        mean, scale = np.asarray(cn['mean'], dtype=np.float64), np.asarray(cn['scale'], dtype=np.float64)
        hs, ds = np.asarray(hn['history_scale'], dtype=np.float64), np.asarray(hn['dimension_scale'], dtype=np.float64)
        if (mean.shape != (basis.modes, 2) or scale.shape != mean.shape or hs.shape != (4,) or ds.shape != (2,)
                or not all(np.isfinite(v).all() for v in (mean, scale, hs, ds))
                or np.any(scale <= 0) or np.any(hs < 1) or np.any(ds < 1)
                or cn.get('roles_used') != ['FIT'] or cn.get('STOP_CAL_AUDIT_used') is not False
                or cn.get('labels_or_risk_used') is not False or hn.get('roles_used') != ['FIT']
                or hn.get('history_centering') is not False or hn.get('future_or_CAL_or_AUDIT_used') is not False):
            raise ValueError('finite FIT-only coefficient and history/static normalization required')
        self._model = copy.deepcopy(model).eval().requires_grad_(False)
        self._schedule = copy.deepcopy(schedule)
        self._basis = basis
        self._mean, self._scale, self._hscale, self._dscale = [v.copy() for v in (mean, scale, hs, ds)]
        self._prediction_type, self._steps = prediction_type, sampling_steps
        self._device = next(self._model.parameters()).device
        if self._schedule.betas.device != self._device:
            raise ValueError('model and schedule must share the explicitly chosen device')
        self._provenance = dict(protocol=PROTOCOL, production_bound=False,
            bindings=copy.deepcopy(provenance or {}), architecture=self._model.architecture_config(),
            prediction_type=prediction_type, DDIM_steps=sampling_steps, DDIM_eta=0., K=1,
            dataset_archives_decoded_by_wrapper=False, risk_estimator_loaded_by_wrapper=False,
            risk_module_imported_by_wrapper=False,
            model_updates_performed=False, generated_output_is_natural_observation=False,
            physical_frame='fixed_t0_canonical_road_not_ego_following_or_GPS')

    @classmethod
    def from_result_binding(cls, binding, device='cpu'):
        if not isinstance(binding, dict) or binding.get('sha256') != RESULT_SHA:
            raise ValueError('only the approved completed natural-prior training result is accepted')
        result = _json(binding)
        if (result.get('protocol') != TRAIN_PROTOCOL or result.get('status') != 'complete'
                or result['checkpoint']['sha256'] != CHECKPOINT_SHA or result['policy']['sha256'] != POLICY_SHA
                or result['data']['sha256'] != DATA_SHA or result.get('prediction_type') != 'v'
                or result.get('risk_labels_in_training') is not False or result.get('simulated_training_futures') is not False):
            raise ValueError('result does not bind the approved history-only velocity prior')
        policy, data = _json(result['policy']), _json(result['data'])
        for path, digest in result['code_sha256'].items():
            _bound(dict(path=path, sha256=digest))
        # torch1.12 has no weights_only mode; authenticate the exact approved
        # internally produced checkpoint hash before pickle deserialization.
        checkpoint = torch.load(_bound(result['checkpoint']), map_location='cpu')
        if (checkpoint.get('protocol') != TRAIN_PROTOCOL or checkpoint['data'] != result['data']
                or checkpoint['policy'] != result['policy'] or checkpoint['code_sha256'] != result['code_sha256']
                or checkpoint['architecture'] != result['architecture'] or checkpoint['prediction_type'] != 'v'
                or checkpoint['basis'] != data['basis'] or checkpoint['seed'] != result['seed']
                or checkpoint['best_epoch'] != result['best_epoch']
                or checkpoint['coefficient_normalizer'] != data['coefficient_normalizer']
                or checkpoint['history_normalizer'] != data['history_normalizer']):
            raise ValueError('EMA checkpoint metadata differs from the authenticated result')
        arch = checkpoint['architecture']
        if (arch['coefficient_dim'] != 2 * data['basis']['modes']
                or any(arch[key] != policy['generator'][key] for key in ('hidden_dim', 'heads', 'layers', 'feedforward_dim'))):
            raise ValueError('checkpoint architecture does not match its basis/training configuration')
        kwargs = {key: arch[key] for key in ('coefficient_dim', 'hidden_dim', 'heads', 'layers', 'feedforward_dim')}
        # Initial random tensors are overwritten by EMA weights. Preserve the
        # caller's CPU RNG instead of consuming its experiment/sampling stream.
        with torch.random.fork_rng(devices=[]):
            model = JointHistoryDenoiser(**kwargs)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.to(torch.device(device))
        schedule = CosineDiffusionSchedule(policy['generator']['diffusion_steps'])
        if schedule.as_dict() != checkpoint['schedule']:
            raise ValueError('trained schedule differs from reconstructed cosine forward process')
        schedule.to(torch.device(device))
        prior = cls(model, schedule, TrajectoryBasis(int(data['basis']['modes'])),
                    _json(data['coefficient_normalizer']), _json(data['history_normalizer']),
                    prediction_type='v', sampling_steps=policy['sampling']['steps'],
                    provenance=dict(result=copy.deepcopy(binding), checkpoint=result['checkpoint'],
                        policy=result['policy'], prepared_data=result['data'],
                        coefficient_normalizer=data['coefficient_normalizer'], history_normalizer=data['history_normalizer'],
                        trained_code_sha256=result['code_sha256'], wrapper_sha256=_sha(__file__)))
        prior._provenance['production_bound'] = True
        return prior

    def describe(self):
        return copy.deepcopy(self._provenance)

    def sample(self, history13physical, dimensions, road_boundaries, ego_mask, *,
               initial_noise=None, seed=None, x0_callback_factory=None):
        """Return F175 and coefficients from one explicit-noise history query.

        Supply exactly one of seed (nonnegative integer) or float32 initial_noise
        [N,K,2]/[1,N,K,2]. The seed uses a local NumPy generator and does not
        consume global NumPy/PyTorch RNG. factory(decoder, num_DDIM_steps) may
        return a DDIM x0 callback; decoder takes [N,K,2] normalized coefficients
        and returns differentiable float64 [175,N,4] physical states. There is
        no risk label or percentile argument on the prior itself.
        """
        h, dims, road = [np.array(v, dtype=np.float64, copy=True) for v in
                         (history13physical, dimensions, road_boundaries)]
        ego = np.array(ego_mask, copy=True)
        if (h.ndim != 3 or h.shape[0] != 13 or h.shape[2] != 4 or h.shape[1] < 3
                or dims.shape != (h.shape[1], 2) or road.ndim != 1 or len(road) < 2
                or ego.shape != (h.shape[1],) or ego.dtype != np.bool_ or ego.sum() != 1
                or not all(np.isfinite(v).all() for v in (h, dims, road))
                or np.any(dims <= 0) or np.any(np.diff(road) <= 0)):
            raise ValueError('complete unpadded physical H13/static geometry and one actual ego required')
        if (initial_noise is None) == (seed is None):
            raise ValueError('supply exactly one initial_noise or seed')
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError('seed must be an explicit integer in [0,2**64)')
        n, k = h.shape[1], self._basis.modes
        if initial_noise is None:
            z_array = np.random.default_rng(seed).standard_normal((1, n, k, 2)).astype(np.float32)
        else:
            value = initial_noise.detach().cpu().numpy() if isinstance(initial_noise, torch.Tensor) else np.asarray(initial_noise)
            if value.shape == (n, k, 2): value = value[None]
            if value.shape != (1, n, k, 2) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('explicit z requires finite float32 [N,K,2] or [1,N,K,2]')
            z_array = np.array(value, copy=True)
        arrays = dict(history=(h[None] / self._hscale).astype(np.float32),
            dimensions=(dims[None] / self._dscale).astype(np.float32),
            road_boundaries=(road[None] / self._hscale[1]).astype(np.float32),
            road_boundary_mask=np.ones((1, len(road)), bool),
            ego_mask=ego[None], agent_mask=np.ones((1, n), bool))
        features = {key: torch.from_numpy(np.ascontiguousarray(value)).to(self._device) for key, value in arrays.items()}
        decoder = _PhysicalDecoder(self._basis, self._mean, self._scale, h[-1], self._device)
        if x0_callback_factory is not None and not callable(x0_callback_factory):
            raise TypeError('optional callback factory must be callable')
        callback = None if x0_callback_factory is None else x0_callback_factory(decoder, self._steps)
        if callback is not None and not callable(callback):
            raise TypeError('callback factory must return a callable or None')
        z = torch.from_numpy(z_array).to(self._device)
        sampled = ddim_sample(self._model, self._schedule, features, z, steps=self._steps,
                              prediction_type=self._prediction_type, x0_callback=callback)
        with torch.no_grad():
            future = decoder(sampled[0]).cpu().numpy().copy()
        coefficients = sampled[0].cpu().numpy().copy()
        if not np.isfinite(future).all() or not np.array_equal(future[0], h[-1]):
            raise RuntimeError('sample failed finite full-horizon/exact observed anchor invariants')
        provenance = self.describe()
        provenance.update(seed=seed, explicit_initial_noise_supplied=initial_noise is not None,
            initial_noise_sha256=hashlib.sha256(z_array.tobytes()).hexdigest(),
            num_agents=n, ego_index=int(np.flatnonzero(ego)[0]),
            callback_factory_supplied=x0_callback_factory is not None, callback_applied=callback is not None,
            candidate_count=1, sample_selection_or_rejection=False,
            denoiser_conditioned_on_history_static_only=True, caller_future_argument_accepted=False,
            external_callback_semantics_not_inspected=True)
        return dict(future=future, coefficients_normalized=coefficients,
                    coefficients_physical=coefficients.astype(np.float64) * self._scale + self._mean,
                    provenance=provenance)
