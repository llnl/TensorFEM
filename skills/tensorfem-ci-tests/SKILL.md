---
name: tensorfem-ci-tests
description: Add or debug TensorFEM CI example cases using ci.yaml and BoBa ci.py. Use when a request asks for CI coverage, case inputs, skip tags, or targeted CI execution in this repository.
---

# TensorFEM CI cases

- Put named invocations under the Makefile executable basename in root `ci.yaml` (`example_advection`, for example). The active CPU job in `.czgitlab-ci.yml` builds debug and release variants and calls the BoBa runner; it does not enumerate individual cases.
- BoBa's `ci.py` reads `inputs`, `environment`, and `skip` from each case. It judges success by exit status. Add a numerical bound to the executable when a case needs to verify accuracy, then pass that bound through `inputs`.
- The runner launches all cases for an executable before waiting for results. Ensure parallel cases do not write the same output files. `example_advection` skips mesh and grid-function output when `TENSORFEM_CI` is defined.
- The runner removes tested binaries and objects unless passed `--keep-binaries`. For a focused local check after building, run `python3 tpl/BoBa/ci.py --tests example_advection --keep-binaries`; add `--debug` for a debug binary.
- Check relevant backend support before adding cases or `skip` tags. The current active GitLab job is CPU; accelerator jobs in `.czgitlab-ci.yml` are templates.
- For runner behavior, inspect `tpl/BoBa/ci.py` and `tpl/BoBa/AGENTS.md`; keep vendored files unchanged. GitLab calls a runner from `TENSORFEM_TPLS_FARM`, which may differ from the vendored copy and is unavailable in this workspace.
