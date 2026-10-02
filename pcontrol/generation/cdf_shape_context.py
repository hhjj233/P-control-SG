"""Frozen H-only calibrated-CDF shape descriptors, never future-conditioned.

65 probabilities on the fixed physical PET grid (0..4 seconds): right CDF at
0..3.9375, then LEFT CDF at 4. The zero and cap atom widths are retained.
This is a sampled descriptor, NOT an exact encoding of between-grid warp knots.
Construction accepts no requested p, observed PET, or generated trajectory.
"""
import numpy as np

from pcontrol.reference.scene_calibration import SceneCDFWarp

GRID_SECONDS = np.linspace(0., 4., 65)
GRID_SECONDS.setflags(write=False)
CONTEXT_DIM = 65
CONTEXT_KEY = 'frozen_cdf_shape'


def _validated(values):
    values = np.array(values, dtype=np.float64, copy=True)
    if (values.shape != (CONTEXT_DIM,) or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1)) or np.any(np.diff(values) < -1e-12)):
        raise ValueError('finite monotone calibrated CDF descriptor required')
    values.setflags(write=False)
    return values


def shape_from_reference(reference):
    """Query an already H-conditioned immutable reference; no future scoring."""
    values = np.asarray(reference.cdf(GRID_SECONDS, side='right')).copy()
    values[-1] = reference.cdf(4., side='left')
    return _validated(values)


def shape_from_masses(masses, row_nodes, count):
    """OOF cache path; caller must bind masses/nodes to the SAME history/fold."""
    warp = SceneCDFWarp('global', np.asarray(row_nodes).reshape(1, -1))
    m = np.asarray(masses).reshape(1, -1)
    values = warp.cdf(m, [count], GRID_SECONDS[None], side='right')[0]
    values[-1] = warp.cdf(m, [count], np.array([4.]), side='left')[0]
    return _validated(values)


def descriptor_contract():
    return dict(version='natural_frozen_H_only_CDF_shape_v1', dimension=CONTEXT_DIM,
                physical_grid_seconds=GRID_SECONDS.tolist(),
                sides='right_at_first64_physical_nodes_left_at_cap',
                zero_atom='descriptor[0]', cap_atom='1-descriptor[-1]',
                between_grid_warp_crossings_exactly_encoded=False,
                depends_on_p_or_observed_PET_or_generated_future=False,
                FIT_source='same_row_recording_excluded_OOF_CDF_and_own_warp',
                deployment_source='frozen_full_FIT_CDF_conditioned_only_on_H_static',
                true_CDF_claimed=False)
