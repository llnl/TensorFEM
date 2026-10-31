---
name: tensorfem-build
description: Build or verify TensorFEM Makefile examples and BoBa CPU, debug, or CI variants. Use for build failures, executable naming, and local run commands in this repository.
---

# TensorFEM Makefile builds

- TensorFEM has no top-level CMake build. Read the target in the root `Makefile`; it includes `tpl/BoBa/Makefile_boba` for compiler, dependencies, backend selection, and suffixes. Use a targeted `make <target>` before a broad rebuild.
- The default local backend is CPU. For example, `make example_advection` produces `example_advection_cpu.out` in this workspace. `BOBA_DEBUG=1` adds `_debug` before `.out`.
- `TENSORFEM_CI=1` changes example behavior for CI. It differs from BoBa's `BOBA_CI=1` strict-build flag. This is good for strict compiler warnings and checks.
- `TENSORFEM_TPLS_FARM` points to the CI dependency farm. Avoid changing vendored `tpl/BoBa/Makefile_boba` for a TensorFEM-only fix.
- After building, run the exact executable and arguments relevant to the change. For CI case additions, follow the [CI test skill](../tensorfem-ci-tests/SKILL.md).
