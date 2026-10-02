"""Numerically guarded reuse adapter for the frozen context-warp experiment.

The experiment implementation/hashes remain unchanged. Four-way floating-point
convex mixtures may exceed a constant endpoint by one ULP even when each
component is valid. This adapter permits only <=1e-12 feasibility roundoff,
reports it, and projects node ordinates back to their mathematically valid
monotone range. It never clips a PET observation or requested percentile.
"""
import numpy as np

from .scene_context_calibration import ContextCDFWarp


class SceneContextCDFPlugin(ContextCDFWarp):
    """Drop-in CDF/rank/inverse/CRPS adapter; not installed in the generator."""

    def stability_report(self, context):
        original = super().row_nodes(context)
        violation = max(0., float(-original.min()), float(original.max()-1.),
                        float(-np.diff(original, axis=1).min()))
        if not np.isfinite(original).all() or violation > 1e-12:
            raise FloatingPointError("materially invalid context mixture; not a roundoff correction")
        polished = np.maximum.accumulate(np.clip(original, 0., 1.), axis=1)
        return polished, dict(maximum_feasibility_roundoff=violation,
                              maximum_node_correction=float(np.max(np.abs(polished-original))),
                              altered_rows=int(np.any(polished != original, axis=1).sum()),
                              observations_or_percentile_requests_modified=False)

    def row_nodes(self, context):
        return self.stability_report(context)[0]
