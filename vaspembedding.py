"""
This package is designed to convert embedding potentials generated in VASP to "embedding integrals" which can be used in PySCF
As of now, only real space conversions of embedding potentials to integrals is supported.
Therefore the selected basis set for metallic species should contain an ECP!
Embedding potentials can only be reliably translated into all electron bases for metals in reciprocal space.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
from pyscf import gto, scf
from pyscf.dft import numint
from pymatgen.io.vasp import Poscar

#constants
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
EV_TO_HARTREE = 1.0 / 27.211386245988
HARTREE_TO_EV = 27.211386245988

#classes
@dataclass
class LatticeInfo:
    """lattice info from POSCAR file"""
    vectors: np.ndarray
    scale: float = 1.0

    @property
    def volume(self) -> float:
        return np.abs(np.linalg.det(self.vectors))


@dataclass
class EmbeddingPotential:
    """container for embpot data"""
    potential: np.ndarray
    lattice: LatticeInfo
    grid_shape: Tuple[int, int, int]
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    source_file: str = ""
    unit: str = "Hartree"

    @property
    def nx(self) -> int:
        return self.grid_shape[0]

    @property
    def ny(self) -> int:
        return self.grid_shape[1]

    @property
    def nz(self) -> int:
        return self.grid_shape[2]

    @property
    def n_points(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def dv_ang(self) -> float:
        """vol per grid point in Angstrom^3"""
        return self.lattice.volume / self.n_points

    @property
    def dv_bohr(self) -> float:
        """vol per grid point in Bohr^3"""
        return self.dv_ang * (ANG_TO_BOHR ** 3)

    def get_grid_coords(self, unit: str = 'bohr') -> np.ndarray:
        """cart coords of grid points"""
        fx = np.arange(self.nx) / self.nx
        fy = np.arange(self.ny) / self.ny
        fz = np.arange(self.nz) / self.nz

        FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')

        coords = (FX[..., np.newaxis] * self.lattice.vectors[0] +
                  FY[..., np.newaxis] * self.lattice.vectors[1] +
                  FZ[..., np.newaxis] * self.lattice.vectors[2])
        coords = coords + self.origin

        if unit.lower() in ['bohr', 'au']:
            coords = coords * ANG_TO_BOHR

        return coords.reshape(-1, 3)

    def pad_periodic(self, nx_repeat: float = 1.0, ny_repeat: float = 1.0,
                     nz_repeat: float = 0.0) -> 'EmbeddingPotential':
        """
        Pad the potential by repeating periodically in x, y, z directions.
        nx_repeat, ny_repeat, nz_repeat are fractional cell repeats added on each side.
        0.5 adds half a unit cell on each side (2x total in that direction).
        1.0 adds one full cell on each side (3x total, original behavior).
        Integer values reproduce the original behavior exactly.
        """
        # Grid points to add on each side
        pad_x = int(np.floor(nx_repeat * self.nx))
        pad_y = int(np.floor(ny_repeat * self.ny))
        pad_z = int(np.floor(nz_repeat * self.nz))

        new_nx = self.nx + 2 * pad_x
        new_ny = self.ny + 2 * pad_y
        new_nz = self.nz + 2 * pad_z

        # Tile enough to cover the padding on both sides, then slice
        tile_x = int(np.ceil(nx_repeat)) * 2 + 1
        tile_y = int(np.ceil(ny_repeat)) * 2 + 1
        tile_z = int(np.ceil(nz_repeat)) * 2 + 1

        tiled = np.tile(self.potential, (tile_x, tile_y, tile_z))

        # Offset into the tiled array where our padded region starts
        ox = (tile_x // 2) * self.nx - pad_x
        oy = (tile_y // 2) * self.ny - pad_y
        oz = (tile_z // 2) * self.nz - pad_z

        new_potential = tiled[ox:ox+new_nx, oy:oy+new_ny, oz:oz+new_nz].copy()

        # Lattice vectors scale with actual grid point count
        new_vectors = self.lattice.vectors.copy()
        new_vectors[0] *= new_nx / self.nx
        new_vectors[1] *= new_ny / self.ny
        new_vectors[2] *= new_nz / self.nz

        new_lattice = LatticeInfo(vectors=new_vectors, scale=self.lattice.scale)

        # Origin shifts by the physical size of the padding added on each side
        new_origin = (self.origin
                      - (pad_x / self.nx) * self.lattice.vectors[0]
                      - (pad_y / self.ny) * self.lattice.vectors[1]
                      - (pad_z / self.nz) * self.lattice.vectors[2])

        print(f"Padded potential: {self.grid_shape} -> ({new_nx}, {new_ny}, {new_nz})")
        print(f"  Repeat: ({nx_repeat}, {ny_repeat}, {nz_repeat}) cells each side")
        print(f"  New origin: [{new_origin[0]:.3f}, {new_origin[1]:.3f}, {new_origin[2]:.3f}] Å")
        print(f"  Grid points: {self.n_points:,} -> {new_nx * new_ny * new_nz:,}")

        return EmbeddingPotential(
            potential=new_potential,
            lattice=new_lattice,
            grid_shape=(new_nx, new_ny, new_nz),
            origin=new_origin,
            source_file=self.source_file,
            unit=self.unit
        )

#reading input files
def read_vasp_potential(extpot_file: str, poscar_file: str,
                        origin: np.ndarray = None,
                        potential_unit: str = 'eV') -> EmbeddingPotential:
    """
    Reads embedding potential and POSCAR info.
    extpot_file:    Path to EXTPOT file
    poscar_file:    Path to POSCAR file of cluster
    origin:         Origin shift in Angstrom
    potential_unit: Unit of potential in file: 'eV' (default) or 'Hartree'
    """
    #read lattice w/ pymatgen
    poscar = Poscar.from_file(poscar_file)
    vectors = poscar.structure.lattice.matrix
    lattice = LatticeInfo(vectors=vectors)

    if origin is None:
        origin = np.zeros(3)

    #read EXTPOT file
    with open(extpot_file, 'r') as f:
        first_line = f.readline().split()
        nx, ny, nz = int(first_line[0]), int(first_line[1]), int(first_line[2])

        values = []
        for line in f:
            values.extend([float(x) for x in line.split()])

    potential = np.array(values).reshape((nz, ny, nx)).transpose(2, 1, 0)
    #VASP potentials are in eV. We need to convert to hartree for PySCF
    if potential_unit.lower() == 'ev':
        potential *= EV_TO_HARTREE

    return EmbeddingPotential(
        potential=potential,
        lattice=lattice,
        grid_shape=(nx, ny, nz),
        origin=origin,
        source_file=extpot_file,
        unit='Hartree'
    )


#compute integrals in parallel
def _worker_init(nthreads):
    import os
    os.environ['OMP_NUM_THREADS'] = str(nthreads)


def _process_batch(args):
    mol_serialized, coords_batch, pot_batch = args
    mol = gto.loads(mol_serialized)
    ao = numint.eval_ao(mol, coords_batch)
    # pot_batch is pre-scaled by dv; keep ao C-contiguous to avoid
    # a transposed intermediate in the DGEMM
    ao_weighted = ao * pot_batch[:, np.newaxis]  # (batch, nao), C-contiguous
    return np.dot(ao_weighted.T, ao)


def _batch_iter(mol_serialized, coords, potential_scaled, batch_size):
    """Generator: streams (mol, coords, pot) tuples to avoid pre-allocating
    all batch args simultaneously in memory."""
    n = len(coords)
    for i0 in range(0, n, batch_size):
        i1 = min(i0 + batch_size, n)
        yield (mol_serialized, coords[i0:i1].copy(), potential_scaled[i0:i1].copy())


def compute_embedding_matrix(mol: gto.Mole, emb_pot: EmbeddingPotential,
                              batch_size: int = 20000,
                              nproc: int = 1,
                              nthreads: int = 1) -> np.ndarray:
    """
    Compute the embedding potential matrix in the AO basis via numerical integration.
    Set nproc to -1 to use all available CPUs, 1 to compute serially.
    nthreads sets the number of OpenMP threads per worker process.
    Total cores used = nproc * nthreads (should not exceed allocated CPUs).
    Batch size can be tuned for better performance but larger batches require more memory.
    """
    import multiprocessing as mp
    import os

    nao = mol.nao_nr()
    coords = emb_pot.get_grid_coords(unit='bohr')
    # Pre-scale potential by dv once here rather than inside every batch
    potential_scaled = emb_pot.potential.ravel() * emb_pot.dv_bohr
    n_points = len(coords)

    print(f"Computing embedding integrals on {n_points:,} grid points...")
    print(f"  Grid: {emb_pot.grid_shape}, NAO: {nao}")

    n_batches = (n_points + batch_size - 1) // batch_size

    if nproc == -1:
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK', None)
        total_cpus = int(slurm_cpus) if slurm_cpus is not None else mp.cpu_count()
        nproc = max(1, total_cpus // nthreads)
    nproc = min(nproc, n_batches)

    if nproc > 1:
        print(f"  Using {nproc} parallel processes, {nthreads} OMP threads each")

        mol_serialized = mol.dumps()

        # imap_unordered streams batches to workers rather than pre-allocating
        # all batch args; chunksize=4 amortizes dispatch overhead without
        # holding many large batches in flight simultaneously
        with mp.Pool(nproc, initializer=_worker_init, initargs=(nthreads,)) as pool:
            results = pool.imap_unordered(
                _process_batch,
                _batch_iter(mol_serialized, coords, potential_scaled, batch_size),
                chunksize=4,
            )
            v_emb = sum(results)

    else:
        v_emb = np.zeros((nao, nao))

        for i0 in range(0, n_points, batch_size):
            i1 = min(i0 + batch_size, n_points)
            ao = numint.eval_ao(mol, coords[i0:i1])
            ao_weighted = ao * potential_scaled[i0:i1, np.newaxis]
            v_emb += np.dot(ao_weighted.T, ao)

    v_emb = 0.5 * (v_emb + v_emb.T)

    print(f"  Done. ||V_emb||_max = {np.abs(v_emb).max():.6e} Ha")

    return v_emb


def compute_embedding_energy(dm: np.ndarray, v_emb: np.ndarray) -> float:
    """Compute embedding energy contribution: E_emb = Tr(D · V_emb)."""
    return np.einsum('ij,ji->', dm, v_emb)


#add to SCF
def add_embedding_potential(mf, v_emb: np.ndarray,
                            inplace: bool = False):
    """
    Add embpot to a PySCF SCF calc
    """
    if not inplace:
        mf = mf.copy()

    original_get_hcore = mf.get_hcore

    def get_hcore_with_embedding(mol=None):
        h = original_get_hcore(mol)
        return h + v_emb

    mf.get_hcore = get_hcore_with_embedding
    mf._v_emb = v_emb

    return mf

#output
def write_embedding_integrals(filename: str, v_emb: np.ndarray):
    np.savetxt(filename, v_emb, fmt='%20.12e')
    print(f"Saved embedding integrals to {filename}")

def load_embedding_integrals(filename: str) -> np.ndarray:
    v_emb = np.loadtxt(filename)
    print(f"Loaded embedding integrals from {filename}, shape: {v_emb.shape}")
    return v_emb
