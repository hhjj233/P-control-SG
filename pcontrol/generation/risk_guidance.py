"""Optional external risk controller; the diffusion backend never imports it.

This is a locally verified active-PET target step, not a global controllability
guarantee or a calibrated safety claim. No search, rejection, or hidden samples.
"""
import numpy as np
import torch

from .trajectory_basis import TrajectoryBasis, basis_matrices


class TorchTrajectoryDecoder:
    def __init__(self, basis, coefficient_normalizer, anchors, device="cpu"):
        self.basis = basis
        bp, bv, _ = basis_matrices(basis.times, basis.modes, basis.horizon)
        self.bp, self.bv = [torch.as_tensor(x, dtype=torch.float64, device=device) for x in (bp, bv)]
        self.times = torch.as_tensor(basis.times, dtype=torch.float64, device=device)
        self.mean, self.scale = [torch.as_tensor(coefficient_normalizer[k], dtype=torch.float64, device=device)
                                for k in ("mean", "scale")]
        self.anchors = torch.as_tensor(anchors, dtype=torch.float64, device=device)
        if (self.anchors.ndim != 2 or self.anchors.shape[1] != 4 or self.mean.shape != (basis.modes, 2)
                or self.scale.shape != self.mean.shape or not bool((self.scale > 0).all())):
            raise ValueError("one complete scene, anchors[N,4] and valid coefficient normalization required")

    def __call__(self, normalized):
        if normalized.shape != (self.anchors.shape[0], self.basis.modes, 2):
            raise ValueError("one unpadded all-actor coefficient tensor required")
        c = normalized.to(torch.float64) * self.scale + self.mean
        xy = (self.anchors[None, :, :2] + self.times[:, None, None] * self.anchors[None, :, 2:]
              + torch.einsum("tk,nkd->tnd", self.bp, c))
        velocity = self.anchors[None, :, 2:] + torch.einsum("tk,nkd->tnd", self.bv, c)
        # Basis integrals vanish at t=0, so this also has zero t0 coefficient Jacobian.
        return torch.cat((xy, velocity), dim=-1)


class ActivePETGuidance:
    def __init__(self, reference, decoder, requested_p, *, strength, total_steps=50, last_steps=20, rms_cap=.15):
        if not 0 <= requested_p <= 1 or strength < 0 or not 0 <= last_steps <= total_steps or rms_cap <= 0:
            raise ValueError("valid target, strength, guidance window and coefficient trust radius required")
        self.reference, self.decoder = reference, decoder
        self.requested_p, self.target = float(requested_p), float(reference.target_pet(requested_p))
        self.strength, self.total_steps, self.last_steps, self.rms_cap = strength, total_steps, last_steps, rms_cap
        self.calls, self.trace = 0, []

    def __call__(self, x0, timesteps, x_t):
        if x0.shape[0] != 1:
            raise ValueError("pilot guidance processes one unpadded scene at a time")
        step = self.calls
        self.calls += 1
        if self.strength == 0 or step < self.total_steps - self.last_steps:
            return x0
        with torch.enable_grad():
            state = x0[0].detach().clone().requires_grad_(True)
            future = self.decoder(state)
            active = self.reference.active_witness_pet(future)
            row = dict(step=step, timestep=int(timesteps[0]), supported=bool(active["supported"]),
                       reason=active["reason"], requested_p=self.requested_p, target_pet_seconds=self.target)
            if not active["supported"]:
                if "exact_score" in active:
                    row["current_pet_seconds"] = active["exact_score"]["pet_seconds"]
                self.trace.append(row)
                return x0
            value = active["value"]
            grad = torch.autograd.grad(value, state)[0]
            norm2 = grad.square().sum()
            if not bool(torch.isfinite(grad).all()) or float(norm2) < 1e-14:
                row.update(supported=False, reason="flat_or_nonfinite_coefficient_gradient")
                self.trace.append(row)
                return x0
            delta = -self.strength * (value.detach() - self.target) * grad / norm2
            rms = delta.square().mean().sqrt()
            delta = delta * torch.clamp(delta.new_tensor(self.rms_cap) / rms.clamp_min(1e-15), max=1.)
            if not bool(torch.isfinite(delta).all()):
                raise FloatingPointError("nonfinite active-PET coefficient update")
            row.update(current_pet_seconds=float(value.detach()), gradient_l2=float(norm2.sqrt()),
                       uncapped_update_RMS=float(rms), applied_update_RMS=float(delta.square().mean().sqrt()),
                       local_oracle_finite_difference_checks=len(active["finite_difference_checks"]))
            self.trace.append(row)
            return (state.detach() + delta.detach())[None].to(x0.dtype)


def interval_p_error(requested_p, rank):
    """Distance to the identified estimated rank interval (atoms are not split)."""
    return float(max(float(rank["p_low"]) - requested_p, requested_p - float(rank["p_up"]), 0.))
