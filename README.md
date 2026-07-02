# Dielectric-bath-embedding-theory-DBET-
########################
Installation
########################


This module must be integrated into the VASP 6 source code.
It has been tested only with version 6.3.2.

Follow the installation procedures for:

VASPsol++: https://github.com/VASPsol/VASPsol

VASPEmbedding: https://github.com/EACcodes/VASPEmbedding

Replace the standard solvation.F file with the solvation.F file provided in this repository.

Use the linear + local solvation model from the original VASPsol implementation by setting:
ISOL = 1

########################
Usage
########################

Add the following flag to your INCAR file:

LSHAPEZ = .TRUE.

When enabled, the code reads a file named S_Z in EXTPOT format.

# Spin-Resolved DBET
Note that SR-DBET was implemented in PySCF, not VASP.

yukawa_spin_pcm.py — Spin-channel PCM using a screened Yukawa kernel instead of bare Coulomb. Decouples PCM from the inner SCF via an outer macro-iteration loop (inner SCF converges under a frozen potential, PCM updates from the converged DM, repeat until ΔV_mat converges). Patches directly onto a UHF/UKS mean-field object.

vasp_embedding.py — Reads VASP EXTPOT/LOCPOT + POSCAR, builds one-electron embedding integrals for a PySCF cluster calculation. Supports periodic padding of the potential grid and interpolation to different resolutions.

pcm_zcavity.py — Restricts a PySCF PCM cavity to one side of a z-interface (e.g. water side of a metal/water slab), with sharp, Fermi, or cosine switching functions.
