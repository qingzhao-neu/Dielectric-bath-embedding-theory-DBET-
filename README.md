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

When enabled, the code reads a file named S_Z in CHG/EXTPOT format.
