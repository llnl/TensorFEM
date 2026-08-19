
# TensorFEM

[![GitHub release](https://img.shields.io/github/release/llnl/tensorfem.svg)](https://github.com/llnl/tensorfem/releases/latest)

The purpose of this research library is to demonstrate how to combine [BoBa](https://github.com/llnl/BoBa) and [MFEM](https://mfem.org/) to solve PDEs.
It is a highly experimental codebase that is likely to change frequently.


## Citation
Please use this citation when citing TensorFEM.
```
  @misc{tensorfemlibrary,
    title  = {TensorFEM: Tensor Finite Element Solvers},
    author = {Guthrey, Pierson and
              Sands, Willian and
              Walton, Steven and
              Hollett, Max and
              Haut, Terry},
    url    = {https://github.com/llnl/tensorfem},
    year   = {2026}
    howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20260817.1}},
    abstractNote = {The purpose of this research library is to demonstrate how to combine the BoBa tensor library and MFEM finite element library to enable tensorized finite element methods.},
    doi = {10.11578/dc.20260817.1}
  }
```

# License

TensorFEM is distributed under the Apache-2.0 with LLVM exception License. See [LICENSE](LICENSE) for
the full terms and [NOTICE](NOTICE) for the LLNL/DOE government notice.

LLNL-CODE-2023234

## Installation

Please see [tpls/README.md](tpls/README.md) to install the third-party libraries (TPLs).

## Examples

Before starting with TensorFEM, we recommend that you are familiar with BoBa and MFEM.

The [assembly example](examples/example_assembly) shows how to form tensorized solvers from a bilinear form.
```
make clean
make example_assembly
./example_assembly_cpu.out
```
Once you have the GLVis plots of the solution, we recommend the following options:
- Flatten the plot (Hold Ctrl -)
- Turn off the light (l)
- F7 for setting the bounds is also helpful
