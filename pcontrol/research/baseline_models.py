"""New, unfrozen supplemental implementations; never mutate primary sources."""
import torch
from torch import nn
from pcontrol.generation.diffusion import _AttentionBlock


class JointPCVAE(nn.Module):
    """Conditional VAE: standard normal per-actor latent, joint decoder.

    No IDs, observed future or fitted PET target enters prior-time sampling.
    Posterior sees natural coefficients only during training/STOP validation.
    """
    def __init__(self, width=128, latent=16):
        super().__init__(); self.latent = latent
        self.history = nn.Sequential(nn.Linear(55, width), nn.SiLU(), nn.Linear(width, width))
        self.context = nn.Sequential(nn.Linear(5, width), nn.SiLU(), nn.Linear(width, width))
        self.encode_blocks = nn.ModuleList([_AttentionBlock(width, 4, 256) for _ in range(2)])
        self.future = nn.Linear(16, width)
        self.posterior_blocks = nn.ModuleList([_AttentionBlock(width, 4, 256) for _ in range(2)])
        self.posterior = nn.Linear(width, 2*latent)
        self.noise = nn.Linear(latent, width)
        self.decode_blocks = nn.ModuleList([_AttentionBlock(width, 4, 256) for _ in range(3)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 16))

    def condition(self, f, p):
        mask = f['agent_mask']; h = f['history']; b, _, n, _ = h.shape
        x = torch.cat((h.permute(0,2,1,3).reshape(b,n,52), f['dimensions'],
                       f['ego_mask'][...,None].float()), -1)
        road = f['road_boundaries']; rm = f['road_boundary_mask']
        context = torch.stack((2*p-1, torch.sin(torch.pi*p), torch.log1p(mask.sum(1).float()),
            road.masked_fill(~rm, torch.inf).min(1).values,
            road.masked_fill(~rm, -torch.inf).max(1).values), -1)
        tokens = self.history(x) + self.context(context)[:,None]
        for block in self.encode_blocks: tokens = block(tokens, mask)
        return tokens

    def decode(self, tokens, noise, mask):
        x = tokens + self.noise(noise)
        for block in self.decode_blocks: x = block(x, mask)
        x = self.output(x).reshape(*mask.shape,8,2)
        return torch.where(mask[...,None,None], x, 0.)

    def sample(self, f, p, noise):
        return self.decode(self.condition(f,p), noise.flatten(2), f['agent_mask'])

    def loss(self, f, p, clean, noise, beta=.01):
        mask=f['agent_mask']; tokens=self.condition(f,p)
        post=tokens+self.future(clean.flatten(2))
        for block in self.posterior_blocks: post=block(post,mask)
        mu, logvar=self.posterior(post).chunk(2,-1); logvar=logvar.clamp(-10.,6.)
        z=mu+torch.exp(.5*logvar)*noise.flatten(2)
        prediction=self.decode(tokens,z,mask)
        denom=mask.sum(1)
        mse=(((prediction-clean)**2).mean((-1,-2))*mask).sum(1)/denom
        kl=(.5*(mu.square()+logvar.exp()-1-logvar).mean(-1)*mask).sum(1)/denom
        return (mse+beta*kl).mean(), mse.mean(), kl.mean()
