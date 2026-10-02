"""Public interface to the released models.

Load the released reference and generator, read scenes prepared by ``pcontrol/tools/prepare_highd.py`` and generate
futures at requested risk percentiles. The computation is the one of the paper's final evaluation: the same network,
calibration map, physical target, sampling-time guidance and constraints, on CPU with the same thread settings.

Example::

    from pcontrol.api import load_reference, load_generator, load_scenes, paper_noise

    reference = load_reference('checkpoints/reference.pt')
    generator = load_generator('checkpoints/generator.pt', reference)
    scene = next(load_scenes('data/scenes/35/complete_scenes.npz'))
    prepared = generator.prepare(scene['features'])
    noise = paper_noise(scene['scene_id'], 0, scene['num_agents'])
    future, info = generator.sample(prepared, 0.9, noise)   # future: (175, N, 4) x, y, vx, vy at 25 Hz
    print(info['pet_seconds'], info['canonical_target_PET_seconds'], info['estimated_rank'])
"""
from pathlib import Path

import numpy as np
import torch

from pcontrol.data.complete_scene_view import complete_eligibility, reference_example, validate_complete_labels
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.generation.trajectory_basis import TrajectoryBasis
from pcontrol.publication_pipeline.final_generation_adapter import (FrozenFinalGenerator, attach_observation_diagnostics,
                                                                   initialize_runtime)
from pcontrol.publication_pipeline.final_reference_suite import FEATURES, FrozenFinalReference
from pcontrol.publication_pipeline.final_validation_protocol import noise_for_history
from pcontrol.reference.history_encoder_ablation import make_ablation
from pcontrol.research import train_guarded_terminal_pair as training
from pcontrol.research.refine_natural_direct_p import state_hash
from pcontrol.time_attention_pipeline import common as c

__all__ = ['FEATURES', 'PAPER_REQUESTS', 'Reference', 'Generator', 'load_reference', 'load_generator', 'load_scenes',
           'paper_noise', 'random_noise', 'attach_observation_diagnostics']

# Requests and noise of the paper's evaluation (five percentiles, three noise draws per history).
PAPER_REQUESTS = (0.1, 0.3, 0.5, 0.7, 0.9)
PAPER_NOISE_SALT = 'final_natural_validation_z_20260916_v1'


class Reference(FrozenFinalReference):
    """Calibrated history-conditioned distribution of the minimum ego--SV PET (the range-query reference).

    ``condition_features(features)`` returns the conditional distribution of one history, with methods such as
    ``cdf(y)``, ``rank(y)`` (the compatible percentile interval of an outcome) and ``score_future(future)``.
    """

    def condition_features(self, features):
        # The final evaluation queried the reference with one CPU thread and sampled with two.
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            return super().condition_features(features)
        finally:
            torch.set_num_threads(previous)


def load_reference(path='checkpoints/reference.pt', device='cpu'):
    """Load the released reference (network weights, feature normalizer and calibration map)."""
    cp = torch.load(path, map_location='cpu', weights_only=False)
    if cp.get('format') != 'pcontrol_reference_v1':
        raise ValueError(f'{path} is not a released reference checkpoint')
    with torch.random.fork_rng(devices=[]):  # initialization must not consume the caller's random stream
        model = make_ablation(cp['arm'])
    if model.architecture_config() != cp['architecture']:
        raise ValueError('reference architecture differs from the checkpoint')
    model.load_state_dict(cp['state_dict'], strict=True)
    model.to(device)
    return Reference(model, cp['normalizer'], c.StableCountWarp.from_dict(cp['calibration']),
                     provenance=dict(checkpoint=str(path)))


class Generator(FrozenFinalGenerator):
    """Percentile-conditioned joint diffusion with sampling-time risk guidance.

    ``prepare(features)`` conditions the reference on a history once. ``sample(prepared, p, noise)`` generates one
    future for the request ``p`` (a risk percentile in (0, 1)) from an initial noise of shape (N, 8, 2) and returns
    the future, of shape (175, N, 4), with a record of its realized PET, percentile interval and target errors.
    """

    def __init__(self, path, reference):
        cp = torch.load(path, map_location='cpu', weights_only=False)
        if cp.get('format') != 'pcontrol_generator_v1':
            raise ValueError(f'{path} is not a released generator checkpoint')
        initialize_runtime()  # two CPU threads, deterministic kernels, math attention (as in the paper)
        self.arm = cp['arm']
        self.training_policy = dict(target_policy=cp['target_policy'])
        self.policy = dict(generation=dict(P_grid=list(PAPER_REQUESTS)))
        with torch.random.fork_rng(devices=[]):
            self.model = training.model_from_checkpoint(cp, self.arm, cp['target_policy'],
                                                        torch.device('cpu')).eval().requires_grad_(False)
        if self.model.architecture_config() != cp['architecture']:
            raise ValueError('generator architecture differs from the checkpoint')
        self.initial_state = state_hash(self.model)
        self.schedule = CosineDiffusionSchedule(cp['diffusion_steps'])
        self.basis = TrajectoryBasis(cp['basis']['K'])
        self.cn, self.hn = cp['coefficient_normalizer'], cp['history_normalizer']
        self.plugin = reference
        self.profile = cp['generation_profile']

    def prepare(self, features):
        """Condition on one history: ``features`` holds history, dimensions, road_boundaries, ego_mask, agent_mask."""
        return self.prepare_case({k: features[k] for k in FEATURES})

    def sample(self, prepared, requested, noise):
        """Generate one future at the requested risk percentile (the paper evaluates 0.1, 0.3, 0.5, 0.7, 0.9)."""
        requested = float(requested)
        if not 0. < requested < 1.:
            raise ValueError('a request is a risk percentile strictly between 0 and 1')
        if requested not in self.policy['generation']['P_grid']:
            self.policy['generation']['P_grid'].append(requested)
        return super().sample(prepared, requested, noise)


def load_generator(path='checkpoints/generator.pt', reference=None):
    """Load the released generator, conditioned through ``reference`` (loaded with ``load_reference``)."""
    return Generator(path, reference if reference is not None else load_reference())


def load_scenes(path):
    """Yield the scenes of a ``complete_scenes.npz`` file written by ``pcontrol/tools/prepare_highd.py``.

    Each scene holds the model features, the recorded future (175, N, 4), its minimum ego--SV PET, the number of
    vehicles, the highD vehicle ids and the scene identity (recording, ego id and start frame).
    """
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    validate_complete_labels(arrays, complete_eligibility(arrays))
    for i in range(len(arrays['scene_id'])):
        example = reference_example(arrays, i)
        lo, hi = map(int, arrays['offsets'][i:i + 2])
        features = example['features']
        features['agent_mask'] = np.ones(hi - lo, bool)
        future = arrays['future_native_agents'][lo:hi].transpose(1, 0, 2).copy()
        if not np.array_equal(future[0], features['history'][-1]):
            raise ValueError('recorded initial state differs from the last history state')
        yield dict(features=features, future_observed=future, natural_PET=example['target'], num_agents=hi - lo,
                   agent_ids=arrays['agent_ids'][lo:hi].copy(), **example['metadata'])


def paper_noise(scene_id, noise_index, num_agents):
    """Initial noise of the paper's evaluation for one scene and noise draw (0, 1 or 2)."""
    return noise_for_history(scene_id, noise_index, int(num_agents), salt=PAPER_NOISE_SALT)[0]


def random_noise(num_agents, seed):
    """Standard normal initial noise of shape (N, 8, 2)."""
    return np.random.default_rng(seed).standard_normal((int(num_agents), 8, 2)).astype(np.float32)
