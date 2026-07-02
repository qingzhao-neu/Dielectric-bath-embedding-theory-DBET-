"""
yukawa_spin_pcm.py - Yukawa-Screened Spin Solvation (Optimized + Outer-Loop PCM)

Key changes vs original:
  1. _compute_v_yukawa and _compute_fock_from_charges now use precomputed
     T_raw matrix (was recomputing the Yukawa kernel from scratch every SCF step).
  2. PCM is decoupled from the inner SCF via a macro-iteration loop:
       - Inner SCF runs to full convergence with a FROZEN Yukawa potential
       - After convergence, PCM is recomputed from the new DM
       - If max(|ΔV_mat|) < pcm_tol, stop; otherwise repeat
       - Typically converges in 3-8 outer iterations
"""

import numpy as np
from scipy.special import erf
from pyscf import gto, df
from pyscf.solvent.pcm import gen_surface, modified_Bondi, get_D_S
from pyscf.dft import gen_grid, numint
import os
from pyscf import scf

__all__ = ['SpinPCM']


class SpinPCM:
    """
    Yukawa-screened spin solvation model (optimized, fully consistent).

    All interactions use the Yukawa kernel exp(-κr)/r:
    - v_grids: spin source → surface potential
    - K matrix: surface ↔ surface
    - Fock: surface → electrons (via direct grid integration, no int3c storage)

    PCM update strategy (macro-iteration):
        Inner SCF runs to full convergence with a FROZEN Yukawa potential.
        After convergence, PCM is recomputed from the converged DM.
        If max(|ΔV_mat|) < pcm_tol, the outer loop stops.
        Typically converges in 3-8 outer iterations.

    Attributes:
        kappa: Inverse screening length (Bohr⁻¹).
        eps: Dielectric constant for spin response.
        spin_channel_mode: 'symmetric' or 'antisymmetric'.
        e_spin: Spin solvation energy after SCF (Hartree).
        q: Surface charges from spin polarization.
        v: Yukawa potential at surface points.
        outer_converged: True if macro-loop converged within max_outer.
        n_outer_iter: Number of macro-iterations taken.
    """

    def __init__(self, mf, kappa=0.5, eps=2.0, grid_level=3,
                 source_prefactor=1.0, verbose=True,
                 spin_channel_mode='symmetric',
                 pcm_tol=1e-6,
                 max_outer=100):
        """
        Args:
            mf: Unrestricted PySCF mean-field object (UHF, UKS, etc.)
            kappa: Inverse Debye screening length (Bohr⁻¹)
            eps: Dielectric constant for spin solvation
            grid_level: DFT grid level for numerical integration (1-9)
            source_prefactor: Scaling prefactor for spin source term
            verbose: Print progress and summary
            spin_channel_mode: 'symmetric' or 'antisymmetric'
            pcm_tol: Outer-loop convergence threshold on max(|ΔV_mat|) in Hartree
            max_outer: Maximum number of macro-iterations
        """
        self._mf = mf
        self.mol = mf.mol
        self.kappa = kappa
        self.eps = eps
        self.grid_level = grid_level
        self.verbose = verbose
        self.source_prefactor = source_prefactor
        self.spin_channel_mode = spin_channel_mode
        self.pcm_tol = pcm_tol
        self.max_outer = max_outer

        # Results
        self.e_spin = 0.0
        self.q = None
        self.v = None
        self.outer_converged = False
        self.n_outer_iter = 0

        # Internal
        self._surface = None
        self._K = None
        self._K_reg = None
        self._K_inv_T = None
        self._R = None
        self._f_eps = None
        self._grids = None
        self._T_raw = None   # T[g, i] = yukawa(r_g, R_i), no weights
        self._ao = None
        self._v_mat_cached = None  # frozen Fock contribution

        self._build()
        self._patch_mf()

    # ─── Build ──────────────────────────────────────────────────────────────

    def _build(self):
        mol = self.mol

        if hasattr(self._mf, 'with_solvent') and self._mf.with_solvent is not None:
            solvent = self._mf.with_solvent
            if not solvent.surface:
                solvent.build()
            self._surface = solvent.surface
        else:
            radii = 1.2 * modified_Bondi
            self._surface = gen_surface(mol, rad=radii, ng=302)

        surf = self._surface
        n_surf = len(surf['grid_coords'])

        if self.verbose:
            print("Building Yukawa K matrix...")
        self._K = self._build_K_yukawa(surf, self.kappa)
        self._K_reg = self._K + 1e-10 * np.eye(n_surf)
        self._K_inv_T = np.linalg.inv(self._K_reg.T)

        if self.eps == float('inf'):
            self._f_eps = 1.0
        else:
            self._f_eps = (self.eps - 1.0) / self.eps

        if 'switch_fun' in surf:
            self._R = -self._f_eps * np.diag(surf['switch_fun'])
        else:
            self._R = -self._f_eps * np.eye(n_surf)

        self._grids = gen_grid.Grids(mol)
        self._grids.level = self.grid_level
        self._grids.verbose = 0
        self._grids.build()

        if self.verbose:
            print("Building grid-surface Yukawa kernel...")
        self._T_raw = self._build_grid_surface_kernel_raw()

        self._ao = numint.eval_ao(mol, self._grids.coords)

        self._load_checkpoint()

        if self.verbose:
            self._print_init_summary(n_surf)

    def _build_K_yukawa(self, surface, kappa):
        coords = surface['grid_coords']
        xi = surface['charge_exp']
        n = len(coords)

        _, S_coulomb = get_D_S(surface, with_S=True, with_D=True)
        coulomb_diag = S_coulomb.diagonal()

        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = coulomb_diag[i] - kappa
            for j in range(i + 1, n):
                r = np.linalg.norm(coords[i] - coords[j])
                xi_ij = xi[i] * xi[j] / np.sqrt(xi[i]**2 + xi[j]**2)
                K[i, j] = erf(xi_ij * r) * np.exp(-kappa * r) / r
                K[j, i] = K[i, j]
        return K

    def _build_grid_surface_kernel_raw(self):
        """
        Build T_raw[g, i] = erf(xi_i * r_gi) * exp(-kappa * r_gi) / r_gi
        WITHOUT integration weights (applied separately as needed).

        This single matrix serves both directions:
          v[i]      = T_raw.T @ (source * weights)   (source → surface)
          V_grid[g] = T_raw @ q                       (charges → grid, for Fock)
        """
        coords = self._grids.coords
        surf_coords = self._surface['grid_coords']
        xi = self._surface['charge_exp']

        n_grid = len(coords)
        n_surf = len(surf_coords)
        chunk_size = 10000
        T = np.zeros((n_grid, n_surf))

        for start in range(0, n_grid, chunk_size):
            end = min(start + chunk_size, n_grid)
            diff = coords[start:end, None, :] - surf_coords[None, :, :]
            dr = np.maximum(np.linalg.norm(diff, axis=2), 1e-10)
            T[start:end, :] = erf(xi[None, :] * dr) * np.exp(-self.kappa * dr) / dr

        return T

    # ─── Checkpoint I/O ──────────────────────────────────────────────────────

    def _load_checkpoint(self):
        """Reconstruct SpinPCM potential from DM stored in mf.chkfile."""
        chkfile = getattr(self._mf, 'chkfile', None)
        if not chkfile or not os.path.isfile(chkfile):
            return
        try:
            mo_coeff = scf.chkfile.load(chkfile, 'scf/mo_coeff')
            mo_occ   = scf.chkfile.load(chkfile, 'scf/mo_occ')
            if mo_coeff is None or mo_occ is None:
                return
            dm = self._mf.make_rdm1(mo_coeff, mo_occ)
            if not (isinstance(dm, np.ndarray) and dm.ndim == 3 and dm.shape[0] == 2):
                return
            e, v_mat, q, v = self._solve(dm[0], dm[1])
            self._v_mat_cached = v_mat
            self.e_spin = e
            self.q = q
            self.v = v
            if self.verbose:
                print(f"[SpinPCM] Warm start from {chkfile}: "
                      f"E_spin = {e * 627.5094:.6f} kcal/mol")
        except Exception as ex:
            if self.verbose:
                print(f"[SpinPCM] WARNING: could not reconstruct from chkfile ({ex}); cold start.")

    # ─── Core solvers ─────────────────────────────────────────────────────────

    def _compute_v_yukawa(self, dm_a, dm_b):
        """
        Compute surface potential vector using precomputed T_raw.
        v = T_raw.T @ (source * weights)
        """
        rho_a = numint.eval_rho(self.mol, self._ao, dm_a)
        rho_b = numint.eval_rho(self.mol, self._ao, dm_b)

        if self.spin_channel_mode == 'antisymmetric':
            source = rho_a - rho_b
        else:
            source = np.abs(rho_a - rho_b)

        return self._T_raw.T @ (self.source_prefactor * source * self._grids.weights)

    def _compute_fock_from_charges(self, q):
        """
        Compute Fock matrix contribution using precomputed T_raw.
        V_grid = T_raw @ q  →  V_mat[μ,ν] = Σ_g ao[g,μ] * w[g] * V_grid[g] * ao[g,ν]
        """
        wV = self._grids.weights * (self._T_raw @ q)
        return self._ao.T @ (wV[:, None] * self._ao)

    def _solve(self, dm_a, dm_b):
        """Solve for surface charges, energy, and Fock contribution."""
        v = self._compute_v_yukawa(dm_a, dm_b)

        b = self._R @ v
        q = np.linalg.solve(self._K_reg, b)

        vK_inv = self._K_inv_T @ v
        q_sym = 0.5 * (q + self._R.T @ vK_inv)

        e_spin = 0.5 * np.dot(q_sym, v)
        v_mat = self._compute_fock_from_charges(q_sym)

        return e_spin, v_mat, q_sym, v

    # ─── SCF patching ────────────────────────────────────────────────────────

    def _patch_mf(self):
        mf = self._mf
        mol = self.mol
        spin_pcm = self

        _get_veff_orig = mf.get_veff
        _energy_elec_orig = mf.energy_elec
        _kernel_orig = mf.kernel

        def get_veff(mol_arg=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
            """Apply the currently frozen v_mat without recomputing PCM."""
            if mol_arg is None:
                mol_arg = mol
            if dm is None:
                dm = mf.make_rdm1()

            veff = _get_veff_orig(mol_arg, dm, dm_last, vhf_last, hermi)

            if (spin_pcm._v_mat_cached is not None
                    and isinstance(dm, np.ndarray)
                    and dm.ndim == 3 and dm.shape[0] == 2):
                if spin_pcm.spin_channel_mode == 'antisymmetric':
                    veff[0] += spin_pcm._v_mat_cached
                    veff[1] -= spin_pcm._v_mat_cached
                else:
                    veff[0] += spin_pcm._v_mat_cached
                    veff[1] += spin_pcm._v_mat_cached

            return veff

        def energy_elec(dm=None, h1e=None, vhf=None):
            e1, e_coul = _energy_elec_orig(dm, h1e, vhf)
            return e1 + spin_pcm.e_spin, e_coul

        def kernel(*args, **kwargs):
            spin_pcm.outer_converged = False

            for outer_it in range(1, spin_pcm.max_outer + 1):
                spin_pcm.n_outer_iter = outer_it

                if spin_pcm.verbose:
                    print(f"\n[SpinPCM] Outer iteration {outer_it}")

                # Inner SCF to full convergence with frozen v_mat
                result = _kernel_orig(*args, **kwargs)

                dm = mf.make_rdm1()
                if not (isinstance(dm, np.ndarray) and dm.ndim == 3 and dm.shape[0] == 2):
                    break

                # Recompute PCM from converged DM
                e, v_mat, q, v = spin_pcm._solve(dm[0], dm[1])
                spin_pcm.e_spin = e
                spin_pcm.q = q
                spin_pcm.v = v

                delta = (np.inf if spin_pcm._v_mat_cached is None
                         else np.max(np.abs(v_mat - spin_pcm._v_mat_cached)))
                spin_pcm._v_mat_cached = v_mat

                if spin_pcm.verbose:
                    print(f"[SpinPCM] ΔV_mat = {delta:.2e}  "
                          f"E_spin = {spin_pcm.e_spin * 627.5094:.6f} kcal/mol")

                if delta < spin_pcm.pcm_tol:
                    spin_pcm.outer_converged = True
                    if spin_pcm.verbose:
                        print(f"[SpinPCM] Converged in {outer_it} outer iterations.")
                    break
            else:
                if spin_pcm.verbose:
                    print(f"[SpinPCM] WARNING: did not converge in "
                          f"{spin_pcm.max_outer} outer iterations.")

            if spin_pcm.verbose:
                spin_pcm._print_summary()
            return result

        mf.get_veff = get_veff
        mf.energy_elec = energy_elec
        mf.kernel = kernel
        mf.spin_pcm = self

    # ─── Utilities ───────────────────────────────────────────────────────────

    def _print_init_summary(self, n_surf):
        n_grid = len(self._grids.coords)
        nao = self.mol.nao
        mem_T_mb = n_grid * n_surf * 8 / 1e6
        mem_ao_mb = n_grid * nao * 8 / 1e6
        print(f"SpinPCM initialized:")
        print(f"  κ = {self.kappa:.3f} Bohr⁻¹ (decay ~ {0.529/self.kappa:.1f} Å)")
        print(f"  ε = {self.eps}, f(ε) = {self._f_eps:.4f}")
        print(f"  spin_channel_mode = {self.spin_channel_mode}")
        print(f"  pcm_tol = {self.pcm_tol:.1e},  max_outer = {self.max_outer}")
        print(f"  Surface points: {n_surf}")
        print(f"  Grid points:    {n_grid}")
        print(f"  Memory (T_raw + ao): {mem_T_mb + mem_ao_mb:.1f} MB")

    def _print_summary(self):
        print(f"\n{'='*50}")
        print("Spin Solvation (Yukawa PCM)")
        print(f"{'='*50}")
        print(f"  mode         = {self.spin_channel_mode}")
        print(f"  outer iters  = {self.n_outer_iter}  converged = {self.outer_converged}")
        print(f"  E_spin = {self.e_spin:12.8f} Ha")
        print(f"         = {self.e_spin * 627.5094:12.4f} kcal/mol")
        if self.q is not None:
            print(f"  Σq     = {self.q.sum():12.6f}")
        print(f"{'='*50}")

    def __getattr__(self, name):
        return getattr(self._mf, name)

    def __setattr__(self, name, value):
        _own = {'_mf', 'mol', 'kappa', 'eps', 'grid_level', 'verbose',
                'e_spin', 'q', 'v', 'source_prefactor', 'spin_channel_mode',
                'pcm_tol', 'max_outer', 'outer_converged', 'n_outer_iter',
                '_surface', '_K', '_K_reg', '_K_inv_T', '_R', '_f_eps',
                '_grids', '_T_raw', '_ao', '_v_mat_cached'}
        if name.startswith('_') or name in _own:
            object.__setattr__(self, name, value)
        else:
            setattr(self._mf, name, value)


# ─── Visualization ────────────────────────────────────────────────────────────

def export_surface(spin_pcm, filename='spin_surface.xyz'):
    if spin_pcm.q is None:
        raise RuntimeError("Run kernel() first")
    coords = spin_pcm._surface['grid_coords'] * 0.529177
    q = spin_pcm.q
    with open(filename, 'w') as f:
        f.write(f"{len(coords)}\n")
        f.write(f"SpinPCM: kappa={spin_pcm.kappa} mode={spin_pcm.spin_channel_mode} "
                f"E={spin_pcm.e_spin:.6f} Ha\n")
        for i, (x, y, z) in enumerate(coords):
            elem = 'He' if q[i] > 0 else 'Ne'
            f.write(f"{elem} {x:12.6f} {y:12.6f} {z:12.6f}\n")
    print(f"Wrote {filename}")


def export_potential_cube(spin_pcm, filename='spin_potential.cube',
                          margin=3.0, resolution=0.2):
    if spin_pcm.q is None:
        raise RuntimeError("Run kernel() first")
    mol = spin_pcm.mol
    q = spin_pcm.q
    surf_coords = spin_pcm._surface['grid_coords']
    xi = spin_pcm._surface['charge_exp']
    kappa = spin_pcm.kappa

    atom_coords = mol.atom_coords()
    margin_bohr = margin / 0.529177
    resolution_bohr = resolution / 0.529177
    all_coords = np.vstack([atom_coords, surf_coords])
    x_min, y_min, z_min = all_coords.min(axis=0) - margin_bohr
    x_max, y_max, z_max = all_coords.max(axis=0) + margin_bohr
    nx = int(np.ceil((x_max - x_min) / resolution_bohr))
    ny = int(np.ceil((y_max - y_min) / resolution_bohr))
    nz = int(np.ceil((z_max - z_min) / resolution_bohr))
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz
    origin = np.array([x_min, y_min, z_min])

    print(f"Building potential grid: {nx}×{ny}×{nz}...")
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    potential = np.zeros((nx, ny, nz))

    for ix in range(nx):
        if ix % 10 == 0:
            print(f"  {100*ix/nx:.0f}%")
        for iy in range(ny):
            grid_points = np.zeros((nz, 3))
            grid_points[:, 0] = x[ix]
            grid_points[:, 1] = y[iy]
            grid_points[:, 2] = z
            diff = grid_points[:, None, :] - surf_coords[None, :, :]
            dr = np.maximum(np.linalg.norm(diff, axis=2), 1e-10)
            potential[ix, iy, :] = erf(xi[None, :] * dr) * np.exp(-kappa * dr) / dr @ q

    comment = (f"SpinPCM V(r) | kappa={kappa} | mode={spin_pcm.spin_channel_mode} | "
               f"E_spin={spin_pcm.e_spin:.6f} Ha")
    _write_cube(mol, filename, origin, dx, dy, dz, nx, ny, nz, potential, comment)
    print(f"Wrote {filename}")


def export_spin_density_cube(spin_pcm, dm_a, dm_b, filename='spin_density.cube',
                              margin=3.0, resolution=0.2):
    mol = spin_pcm.mol
    atom_coords = mol.atom_coords()
    margin_bohr = margin / 0.529177
    resolution_bohr = resolution / 0.529177
    x_min, y_min, z_min = atom_coords.min(axis=0) - margin_bohr
    x_max, y_max, z_max = atom_coords.max(axis=0) + margin_bohr
    nx = int(np.ceil((x_max - x_min) / resolution_bohr))
    ny = int(np.ceil((y_max - y_min) / resolution_bohr))
    nz = int(np.ceil((z_max - z_min) / resolution_bohr))
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz
    origin = np.array([x_min, y_min, z_min])

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    grid_coords = np.array(np.meshgrid(x, y, z, indexing='ij')).reshape(3, -1).T
    ao = numint.eval_ao(mol, grid_coords)
    spin_dens = (numint.eval_rho(mol, ao, dm_a) - numint.eval_rho(mol, ao, dm_b)).reshape((nx, ny, nz))
    _write_cube(mol, filename, origin, dx, dy, dz, nx, ny, nz, spin_dens,
                comment="Spin density m(r) = rho_alpha - rho_beta (signed)")
    print(f"Wrote {filename}")


def _write_cube(mol, filename, origin, dx, dy, dz, nx, ny, nz, data, comment=""):
    with open(filename, 'w') as f:
        f.write("Cube file from SpinPCM\n")
        f.write(f"{comment}\n")
        natom = mol.natm
        f.write(f"{natom:5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}\n")
        f.write(f"{nx:5d} {dx:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        f.write(f"{ny:5d} {0.0:12.6f} {dy:12.6f} {0.0:12.6f}\n")
        f.write(f"{nz:5d} {0.0:12.6f} {0.0:12.6f} {dz:12.6f}\n")
        for ia in range(natom):
            Z = mol.atom_charge(ia)
            c = mol.atom_coord(ia)
            f.write(f"{int(Z):5d} {float(Z):12.6f} {c[0]:12.6f} {c[1]:12.6f} {c[2]:12.6f}\n")
        count = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    f.write(f"{data[ix, iy, iz]:13.5e}")
                    count += 1
                    if count % 6 == 0:
                        f.write("\n")
                if count % 6 != 0:
                    f.write("\n")
                    count = 0


def export_all(spin_pcm, dm_a, dm_b, prefix='spinpcm', margin=3.0, resolution=0.3):
    export_surface(spin_pcm, f"{prefix}_surface.xyz")
    export_potential_cube(spin_pcm, f"{prefix}_potential.cube", margin, resolution)
    export_spin_density_cube(spin_pcm, dm_a, dm_b, f"{prefix}_spindens.cube", margin, resolution)
