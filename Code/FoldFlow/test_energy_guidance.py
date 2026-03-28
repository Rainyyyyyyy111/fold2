import torch
import numpy as np
import sys
import os

# Ensure FoldFlow modules can be imported
sys.path.append(r"D:\Paper\Code\FoldFlow")
from FoldFlow.so3.so3_helpers import expmap, pt_to_identity

def random_so3(shape):
    Z = torch.randn(*shape, 3, 3)
    Q, R = torch.linalg.qr(Z)
    sign = torch.sign(torch.linalg.det(Q))
    Q = Q * sign[..., None, None]
    return Q

def main():
    print("Setting up local Energy Guidance Test on SE(3)...")
    batch_size = 2
    num_residues = 16
    dt = 0.01

    # 1. Mock Rotation at time t (must be valid SO(3))
    rot_t = random_so3((batch_size, num_residues))

    # 2. Mock model output vector field (must be in Tangent space T_R SO(3))
    # v_t = R * A, where A is in Lie algebra so(3)
    skew = torch.randn(batch_size, num_residues, 3, 3)
    skew = 0.5 * (skew - skew.transpose(-1, -2))
    v_t_model = rot_t @ skew

    # 3. Setup gradient tracking for Energy guidance
    rot_t_tensor = rot_t.clone().detach().requires_grad_(True)

    # 4. Define a mock Energy Function (e.g. attractive potential to a target structure)
    R_target = random_so3((batch_size, num_residues))
    # E(R) = Frobenius norm squared difference
    energy = torch.sum((rot_t_tensor - R_target)**2)
    print(f"Initial Energy: {energy.item():.4f}")

    # 5. Compute ambient gradient: \nabla_R E(R)
    grad_E = torch.autograd.grad(energy, rot_t_tensor)[0]

    # 6. Manifold-Constrained Tangent Projection: Proj_{T_R SO(3)}(grad_E)
    # The projected gradient on the manifold is R * (R^T G - G^T R)/2
    R_T_G = torch.transpose(rot_t_tensor, -2, -1) @ grad_E
    skew_G = 0.5 * (R_T_G - torch.transpose(R_T_G, -2, -1))
    grad_E_proj = rot_t_tensor @ skew_G

    # 7. Apply guidance to the vector field
    alpha = 10.0 # Guidance scale
    guided_v_t = v_t_model + alpha * grad_E_proj

    # 8. Perform the integration step (using FoldFlow's native expmap)
    # Note: reverse integration means we step backwards in time, so we subtract
    perturb = -guided_v_t * dt
    rot_t_1 = expmap(rot_t_tensor.double(), perturb.double())

    # 9. Verification
    # Check if the output is still strictly on the SO(3) manifold
    I_target = torch.eye(3).double().unsqueeze(0).unsqueeze(0).expand(batch_size, num_residues, 3, 3)
    ortho_error = torch.max(torch.abs(rot_t_1.transpose(-1, -2) @ rot_t_1 - I_target))
    det_error = torch.max(torch.abs(torch.linalg.det(rot_t_1) - 1.0))
    
    # Check if energy decreased (ignoring the model v_t drift for this check, just looking at gradient direction)
    # To strictly test energy descent, let's step ONLY using the guidance vector field
    perturb_energy_only = -(alpha * grad_E_proj) * dt
    rot_t_1_energy_only = expmap(rot_t_tensor.double(), perturb_energy_only.double())
    new_energy = torch.sum((rot_t_1_energy_only.float() - R_target)**2)

    print("\n--- Validation Results ---")
    print(f"Max Orthogonality Error (R^T R - I): {ortho_error.item():.2e}")
    print(f"Max Determinant Error (|det(R) - 1|): {det_error.item():.2e}")
    print(f"Energy after step: {new_energy.item():.4f} (Decreased: {new_energy < energy})")

    if ortho_error < 1e-6 and det_error < 1e-6:
        print("\nSUCCESS: The Manifold-Constrained Tangential Projection keeps the structure strictly on SE(3)!")
    else:
        print("\nFAILURE: Manifold constraint violated.")

if __name__ == '__main__':
    main()
