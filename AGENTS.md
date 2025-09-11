# Repository Guidelines

## Project Structure & Module Organization
- Core Python scripts: `task_stacking.py`, `ur10_table_stacking.py`, `table_setup.py` (entry points/demos).
- Isaac Sim snapshot (read-only reference) under `exts/isaacsim/`:
  - `core/` — utilities, prims, API layers, and tests under `core/**/tests`.
  - `robot/manipulators/` — controllers, grippers, examples, and OGN nodes.
  - `robot/manipulators/tests` and `core/api/**/tests` — example/unit tests.
- Assets/examples: icons (`*.svg`) and sample USD files for tests.

## Build, Test, and Development Commands
- Run demos: `python task_stacking.py` (see scripts above for variants).
- Run tests for the copied Isaac modules: `pytest exts/isaacsim -q`.
- Run a focused test file: `pytest exts/isaacsim/core/api/tests/test_world.py -q`.
Notes: The repo is pure Python; no build step. Running demos assumes a working Isaac Sim Python environment on your machine.

## Coding Style & Naming Conventions
- Python 3.x, PEP 8, 4-space indentation, max line length ~100–120.
- Filenames/modules: `snake_case.py`; Classes: `PascalCase`; Functions/vars: `snake_case`.
- Prefer explicit imports from `exts/isaacsim/...` for clarity.
- Add docstrings for public functions; include brief rationale for non-obvious math/transforms.

## Testing Guidelines
- Use `pytest`; place tests alongside modules in `tests` folders or `ogn/tests` where applicable.
- Test names: files `test_*.py`; functions `test_<unit_of_behavior>`.
- Keep demos smoke-testable (e.g., fast paths, flags to reduce runtime). Aim to keep unit tests <1s each.

## Commit & Pull Request Guidelines
- Commits: imperative mood (“Add…”, “Fix…”); group logical changes. Prefix types when helpful (e.g., `feat:`, `fix:`, `refactor:`) — this matches prior history.
- PRs: include a concise description, steps to reproduce/run, and screenshots or short clips if behavior changes. Link related issues. Note any Isaac Sim version assumptions.

## Agent-Specific Tips
- Validate imports using repository-relative paths used by IDEs (see prior commits improving import resolution).
- Keep changes minimal to avoid diverging from the reference `exts/isaacsim` snapshot; prefer adapters/wrappers in top-level scripts over editing reference copies.

