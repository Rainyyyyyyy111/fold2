import torch
import torch.nn as nn
import numpy as np

# Ensure FoldFlow modules can be imported if used stand-alone
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foldflow.so3.so3_helpers import pt_to_identity

class BinderEnergyGuidance(nn.Module):
    """
    Computes energy gradients for Zero-Shot Binder Design.
    Provides methods to project these gradients onto the SE(3) manifold tangent spaces.
    """
    def __init__(self, target_coords, attract_scale=10.0, repel_scale=5.0, clash_dist=3.0, contact_dist=8.0):
        super().__init__()
        # target_coords: [N_target, 3] representing the CA coordinates of the target (e.g., Abeta42)
        self.register_buffer('target_coords', torch.tensor(target_coords, dtype=torch.float32))
        self.attract_scale = attract_scale
        self.repel_scale = repel_scale
        self.clash_dist = clash_dist
        self.contact_dist = contact_dist

    def compute_energy(self, binder_trans):
        """
        Compute pseudo-energy for the current state of the binder.
        binder_trans: [Batch, N_binder, 3] translation vectors (CA coords)
        """
        # Calculate pairwise distances between binder residues and target residues
        # binder_trans: [B, N, 1, 3]
        # target_coords: [1, 1, M, 3]
        dists = torch.cdist(binder_trans, self.target_coords.unsqueeze(0).expand(binder_trans.shape[0], -1, -1))
        
        # 1. Attractive Energy: Encourage binder to be close to the target, but not too close.
        # We want at least some residues to be within contact_dist.
        # Using soft-min to find the closest distances smoothly.
        min_dists_per_binder = torch.min(dists, dim=-1)[0] # [B, N]
        
        # We want the *overall* structure to approach, penalize large distances to surface
        # Let's say we want the 20% closest residues to be very close
        k = max(1, int(0.2 * binder_trans.shape[1]))
        closest_k_dists = torch.topk(min_dists_per_binder, k, dim=-1, largest=False)[0]
        attract_energy = torch.mean(closest_k_dists, dim=-1)

        # 2. Repulsive Energy (Steric Clash): Heavily penalize atoms closer than clash_dist
        # E_repel = sum( max(0, clash_dist - dist)^2 )
        clash_penalties = torch.relu(self.clash_dist - dists)
        repel_energy = torch.sum(clash_penalties**2, dim=(-2, -1))

        total_energy = self.attract_scale * attract_energy + self.repel_scale * repel_energy
        return total_energy

    def get_guidance_gradients(self, rot_t, trans_t):
        """
        Calculates gradients w.r.t translations and projects rotation gradients onto the SO(3) tangent space.
        
        rot_t: [Batch, N, 3, 3]
        trans_t: [Batch, N, 3]
        
        Returns:
            guided_v_rot: [Batch, N, 3, 3] (Tangent space projection of dE/dRot)
            guided_v_trans: [Batch, N, 3] (dE/dTrans)
        """
        with torch.enable_grad():
            trans_tensor = trans_t.clone().detach().requires_grad_(True)
            rot_tensor = rot_t.clone().detach().requires_grad_(True)
            
            # Currently, basic binder guidance only depends on translations (CA atoms).
            # To make it dependent on rotations (e.g. side-chain orientation), 
            # we would compute CB atom positions using rot_tensor. 
            # For simplicity in this base proof-of-concept, we guide translations.
            
            energy = self.compute_energy(trans_tensor)
            
            # Sum energy over batch for parallel gradient computation
            sum_energy = torch.sum(energy)
            
            grad_trans = torch.autograd.grad(sum_energy, trans_tensor)[0]
            
            # Since energy here only depends on trans, grad_rot will be zero.
            # (In a full implementation using all atoms, grad_rot would be non-zero).
            grad_rot = torch.zeros_like(rot_tensor)
            
            # --- SO(3) Manifold Projection ---
            # Project grad_rot onto the Tangent Space T_R SO(3)
            # Projection formula: R * Skew(R^T * G)
            R_T_G = torch.transpose(rot_tensor, -2, -1) @ grad_rot
            skew_G = 0.5 * (R_T_G - torch.transpose(R_T_G, -2, -1))
            proj_grad_rot = rot_tensor @ skew_G
            
        return proj_grad_rot, grad_trans
