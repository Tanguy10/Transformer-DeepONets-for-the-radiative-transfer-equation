"""Build the training dataset (.pt) from the raw FVDOM solver output (.h5).

    python src/prepare_data.py --h5 /path/to/smallPoolFire2D.h5 --out data/dataset_train_full_phi.pt

Reads every `time_*` group of the H5 file and writes a single dict of tensors:

  dict_data  supervised pairs -- inputs (T, kappa) per cell, target I on the full
             22500 cells x 16 ordinates grid, and the shared query coordinates
             y = (x, y, phi), 3 columns;
  dict_pde   10000 collocation points drawn uniformly in (x, y, phi), with the RTE
             source terms read at the nearest cell;
  dict_bc    1000 boundary points, each on the INCOMING half-circle of its wall.

The file is ~10 GB and is what every training run loads. Expect a few minutes and
a comparable amount of RAM.
"""
import argparse
import h5py
import torch
import numpy as np
import os
import math
from scipy.spatial import KDTree

from utils import dirs_from_phi, ordinate_phis, DEFAULT_DATASET, N_DIRS_DATA, SOLVER_H5


def main(h5_path=SOLVER_H5, output_pt_path=DEFAULT_DATASET):

    print("=== BUILDING THE CONTINUOUS, MESH-FREE DATASET -- (x, y, phi) queries ===")

    with h5py.File(h5_path, 'r') as f:
        cell_centers = f['mesh/cell_centers'][:]
        x_real = cell_centers[:, 0]
        y_real = cell_centers[:, 1]

        x_min, x_max = x_real.min(), x_real.max()
        y_min, y_max = y_real.min(), y_real.max()
        print(f" -> Domain bounds: X = [{x_min:.4f}, {x_max:.4f}] | Y = [{y_min:.4f}, {y_max:.4f}]")

        tree = KDTree(np.stack([x_real, y_real], axis=1))

        time_groups = sorted([key for key in f.keys() if key.startswith('time_')])
        num_snapshots = len(time_groups)

        list_kappa = []
        list_T = []
        list_I = []
        for t_grp in time_groups:
            list_kappa.append(f[f"{t_grp}/aLambda_0"][:])
            list_T.append(f[f"{t_grp}/T"][:])
            list_I.append(f[f"{t_grp}/I"][:])

    kappa_all = np.array(list_kappa, dtype=np.float32)
    T_all = np.array(list_T, dtype=np.float32)
    I_all = np.array(list_I, dtype=np.float32)

    Ib_all = (5.670374419e-8 / math.pi) * T_all ** 4
    u1_all_spatial = kappa_all * Ib_all
    u2_all_spatial = -kappa_all

    N_cells = 22500
    N_dirs = N_DIRS_DATA
    phi_discrete = ordinate_phis(N_dirs)
    dir_x_discrete = dirs_from_phi(phi_discrete)[:, 0]

    print("\n=== SOLVER VALUES AT THE BOUNDARY (physics sanity check) ===")
    idx_paroi_gauche = np.where(np.abs(x_real - x_min) < 1e-3)[0]
    if len(idx_paroi_gauche) > 0:
        c_idx = idx_paroi_gauche[len(idx_paroi_gauche)//2]
        print(f"Boundary cell {c_idx} at x={x_real[c_idx]:.3f}, y={y_real[c_idx]:.3f}:")
        for m in range(N_dirs):
            nature = "incoming (from outside)" if dir_x_discrete[m] > 0.05 else (
                     "outgoing (from the fire)" if dir_x_discrete[m] < -0.05 else "grazing")
            print(f"  > direction {m} (φ={phi_discrete[m]:.3f}, sin φ={dir_x_discrete[m]:+.2f}) "
                  f"| {nature:<22} | I_FVDOM = {I_all[0, c_idx, m]:.4e}")

    print("\nGenerating the PDE collocation points, continuous in space and direction...")
    np.random.seed(42)
    m_int = 10000

    x_pde_rand = np.random.uniform(x_min, x_max, m_int)
    y_pde_rand = np.random.uniform(y_min, y_max, m_int)
    phi_pde_rand = np.random.uniform(0.0, 2 * math.pi, m_int)

    _, pde_nearest_cells = tree.query(np.stack([x_pde_rand, y_pde_rand], axis=1))

    y_pde = torch.tensor(np.stack([x_pde_rand, y_pde_rand, phi_pde_rand], axis=1), dtype=torch.float32)
    u1_eval = torch.from_numpy(u1_all_spatial[:, pde_nearest_cells])
    u2_eval = torch.from_numpy(u2_all_spatial[:, pde_nearest_cells])

    print("Generating the BC points on the geometric boundary lines...")
    m_b = 1000
    bc_points = []
    bc_sides = []
    points_per_face = m_b // 4

    # INCOMING half-circle for each face, in phi (= atan2(mu_x, mu_y), from +y
    # clockwise): left sin phi>0, right sin phi<0, bottom cos phi>0, top cos phi<0.
    for _ in range(points_per_face):

        p = np.random.uniform(0.0, math.pi)
        bc_points.append([x_min, np.random.uniform(y_min, y_max), p])
        bc_sides.append("left")

        p = np.random.uniform(math.pi, 2*math.pi)
        bc_points.append([x_max, np.random.uniform(y_min, y_max), p])
        bc_sides.append("right")

        p = np.random.uniform(-math.pi/2, math.pi/2) % (2*math.pi)
        bc_points.append([np.random.uniform(x_min, x_max), y_min, p])
        bc_sides.append("bottom")

        p = np.random.uniform(math.pi/2, 3*math.pi/2)
        bc_points.append([np.random.uniform(x_min, x_max), y_max, p])
        bc_sides.append("top")

    y_bc = torch.tensor(bc_points, dtype=torch.float32)

    sigma_sb = 5.670374419e-8
    T_bc = 300.0
    I_bc_val = sigma_sb / math.pi * (T_bc ** 4)

    I_target_bc = torch.full((num_snapshots, m_b), I_bc_val, dtype=torch.float32)

    print("\nPreparing the full supervised data (no subsampling)...")

    u1_in = torch.from_numpy(T_all)
    u2_in = torch.from_numpy(kappa_all)


    x_rep = np.repeat(x_real, N_dirs)
    y_rep = np.repeat(y_real, N_dirs)

    phi_rep = np.tile(phi_discrete, N_cells)

    y_data = torch.tensor(np.stack([x_rep, y_rep, phi_rep], axis=1), dtype=torch.float32)

    I_target_data = torch.from_numpy(I_all.reshape(num_snapshots, -1))

    print(f" -> y_data shape: {y_data.shape}")
    print(f" -> I_target_data shape: {I_target_data.shape}")

    dataset_pinn = {
        'dict_pde': {'u1_in': u1_in, 'u2_in': u2_in, 'u1_eval': u1_eval, 'u2_eval': u2_eval, 'y': y_pde},
        'dict_bc': {'u1_in': u1_in, 'u2_in': u2_in, 'I_target': I_target_bc, 'y': y_bc},
        'dict_data': {'u1_in': u1_in, 'u2_in': u2_in, 'I_target': I_target_data, 'y': y_data}
    }

    os.makedirs(os.path.dirname(os.path.realpath(output_pt_path)) or ".", exist_ok=True)
    torch.save(dataset_pinn, output_pt_path)
    print(f"\n[OK] FULL dataset saved to: {output_pt_path}")
    print("File size: {:.2f} GB".format(os.path.getsize(output_pt_path) / (1024**3)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--h5", default=SOLVER_H5,
                   help="Raw FVDOM solver output (or set RTE2D_SOLVER_H5).")
    p.add_argument("--out", default=DEFAULT_DATASET, help="Dataset .pt to write.")
    a = p.parse_args()
    main(a.h5, a.out)
