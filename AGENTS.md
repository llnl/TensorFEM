# TensorFEM repository guidance

## Build system

- TensorFEM has no top-level `CMakeLists.txt`; its examples are wired manually in the root `Makefile`. That file includes `tpl/BoBa/Makefile_boba`, which selects the compiler, backend, flags, and executable suffix. `make all` includes `example_advection`; `make example_advection` builds only that example.
- The default local CPU executable is `example_advection_cpu.out`. `BOBA_DEBUG=1` produces `example_advection_cpu_debug.out`. CI sets `TENSORFEM_CI=1` and builds both debug and release CPU variants.
- For TensorFEM CI-dependent C++ behavior, use `tensor_fem::is_tensorfem_ci()` with `if constexpr`. Keep preprocessor selection inside that helper, following BoBa's `is_ci_mode()` pattern.
- Consult [skills/tensorfem-build/SKILL.md](skills/tensorfem-build/SKILL.md) for targeted build and variant checks. `tpl/BoBa/AGENTS.md` is a reference for BoBa internals; keep vendored `tpl/` files unchanged for TensorFEM tasks.

## CI test system

- Add executable cases under the executable name in `ci.yaml`; the key must match a Makefile-built executable basename. The active `cpu_build_and_test` job in `.czgitlab-ci.yml` builds debug and release variants with `make all` and invokes BoBa's `ci.py` for each. The accelerator job definitions are currently hidden GitLab templates (their names begin with `.`). Individual cases do not belong in the GitLab job script.
- A `ci.yaml` case can supply `inputs`, `environment`, and backend `skip` tags. The runner checks process exit status, so numerical acceptance checks must make the executable fail when a result is outside its limit.
- BoBa's runner starts cases for one executable concurrently. Cases must not write the same output files; `example_advection` suppresses its mesh and grid-function files when compiled with `TENSORFEM_CI`.
- By default, `ci.py` removes each tested executable and object file after its cases finish. The vendored `tpl/BoBa/ci.py` supports `--tests example_advection --keep-binaries` for a targeted local run, with `--debug` for the debug suffix. GitLab invokes a BoBa runner from the configured TPL farm; that path is unavailable in this workspace, so the vendored runner is a local proxy rather than proof of the farm runner's exact version.
- Consult [skills/tensorfem-ci-tests/SKILL.md](skills/tensorfem-ci-tests/SKILL.md) before adding or changing cases. See `tpl/BoBa/skills/boba-test-authoring/SKILLS.md` for the upstream convention.
