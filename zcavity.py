"""
This package is designed to construct the cavity for solvation only beneath a given coordinate along the z-axis
The dielectric constant changes instantly across this z-value (no smooth transition to the vaccuum region)
Only can be implemented in C_PCM
"""
#we need numpy & pyscf
import numpy as np
from pyscf.solvent.pcm import get_F_A, get_D_S, PI, modified_Bondi

__all__ = ['apply_z_cavity', 'export_xyz', 'export_cube']

#constants
BOHR_TO_ANG = 0.529177
ANG_TO_BOHR = 1.8897259886


def apply_z_cavity(pcm_obj, z_sol=0.0, unit='Angstrom'):
    """
    This just creates removes the cavity in the vaccuum region from the cavity built in PCM
    z_sol defines the z coordinate where we define the surface (where we want to separate vaccuum & solvation)
    """
    if unit.lower() in ['angstrom', 'ang', 'a']:
        z_sol = z_sol * ANG_TO_BOHR

    #build surface if it doesn't exist
    if not pcm_obj.surface:
        pcm_obj.build()

    #store original surface
    if 'switch_fun_orig' not in pcm_obj.surface:
        pcm_obj.surface['switch_fun_orig'] = pcm_obj.surface['switch_fun'].copy()
        pcm_obj.surface['area_orig'] = pcm_obj.surface['area'].copy()

    #active only below z_sol
    z = pcm_obj.surface['grid_coords'][:, 2]
    z_mod = np.where(z < z_sol, 1.0, 0.0)

    #apply modulation
    pcm_obj.surface['switch_fun'] = pcm_obj.surface['switch_fun_orig'] * z_mod
    pcm_obj.surface['area'] = pcm_obj.surface['area_orig'] * z_mod
    pcm_obj.surface['z_modulation'] = z_mod
    pcm_obj.surface['z_sol'] = z_sol

    #rebuild C-PCM matrices
    _rebuild_matrices(pcm_obj)

    return pcm_obj


def _rebuild_matrices(pcm_obj):
    """
    This just rebuilds the C-PCM matrices after we modify the cavity
    """
    F, A = get_F_A(pcm_obj.surface)
    D, S = get_D_S(pcm_obj.surface, with_S=True, with_D=True)

    eps = pcm_obj.eps
    f_eps = (eps - 1.) / eps
    K = S
    R = -f_eps * np.eye(S.shape[0])

    pcm_obj._intermediates.update({'S': S, 'D': D, 'A': A, 'K': K, 'R': R, 'f_epsilon': f_eps})


def export_xyz(pcm_obj, filename='cavity.xyz'):
    """
    Export cavity surface points as an XYZ file
    Active points (below z_sol) are written as Ne, inactive as He
    """
    coords = pcm_obj.surface['grid_coords'] * BOHR_TO_ANG
    swf = pcm_obj.surface['switch_fun']

    with open(filename, 'w') as f:
        f.write(f"{len(coords)}\nPCM cavity (switch_fun in comment)\n")
        for i, (x, y, z) in enumerate(coords):
            sym = 'Ne' if swf[i] > 0.5 else 'He'
            f.write(f"{sym} {x:12.6f} {y:12.6f} {z:12.6f}  # {swf[i]:.4f}\n")

    print(f"Wrote {filename}")


def export_cube(pcm_obj, filename='cavity.cube', spacing=0.3):
    """
    Export cavity as a Gaussian cube file
    """
    mol = pcm_obj.mol
    atom_coords = mol.atom_coords(unit='B')
    charges = mol.atom_charges()  # true atomic numbers
    radii = np.array([pcm_obj.vdw_scale * modified_Bondi[q] for q in charges])

    #grid
    pad = 4.0
    mins = atom_coords.min(axis=0) - pad - max(radii)
    maxs = atom_coords.max(axis=0) + pad + max(radii)
    ns = ((maxs - mins) / spacing).astype(int) + 1

    axes = [np.linspace(mins[i], maxs[i], ns[i]) for i in range(3)]

    #build cavity: 1 = solvent region, 0 = inside solute
    data = np.ones(ns)
    for i, xi in enumerate(axes[0]):
        for j, yj in enumerate(axes[1]):
            for k, zk in enumerate(axes[2]):
                for ia in range(mol.natm):
                    if np.linalg.norm([xi, yj, zk] - atom_coords[ia]) < radii[ia]:
                        data[i, j, k] = 0.0
                        break

    #apply z-cutoff
    if 'z_sol' in pcm_obj.surface:
        z_sol = pcm_obj.surface['z_sol']
        for k, zk in enumerate(axes[2]):
            if zk >= z_sol:
                data[:, :, k] = 0.0

    #write cube
    with open(filename, 'w') as f:
        f.write("PCM Cavity\n1=solvent, 0=solute\n")
        f.write(f"{mol.natm:5d} {mins[0]:12.6f} {mins[1]:12.6f} {mins[2]:12.6f}\n")
        for i in range(3):
            v = [0., 0., 0.]
            v[i] = spacing
            f.write(f"{ns[i]:5d} {v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}\n")
        for ia in range(mol.natm):
            f.write(f"{charges[ia]:5d} {0.:12.6f} {atom_coords[ia, 0]:12.6f} "
                    f"{atom_coords[ia, 1]:12.6f} {atom_coords[ia, 2]:12.6f}\n")
        cnt = 0
        for i in range(ns[0]):
            for j in range(ns[1]):
                for k in range(ns[2]):
                    f.write(f"{data[i, j, k]:13.5e}")
                    cnt += 1
                    if cnt % 6 == 0:
                        f.write("\n")
        if cnt % 6:
            f.write("\n")

    print(f"Wrote {filename} ({ns[0]}x{ns[1]}x{ns[2]})")