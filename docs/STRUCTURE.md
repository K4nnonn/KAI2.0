# Planned Structure (Shadow Only)

This is the intended target layout for the clean repo. No moves yet.

Proposed layout:

- apps/web -> from `repo_shadow/kai-frontend`
- services/api -> from `repo_shadow/kai-backend-complete`
- services/worker -> extracted from `repo_shadow/kai-backend-complete/jobs_worker.py`
- services/llm -> from `repo_shadow/kai-llm-shim` or `repo_shadow/llama-build2`
- tests/e2e -> from `repo_shadow/playwright-run`
- infra/docker -> Dockerfiles (`repo_shadow/kai-backend-complete/Dockerfile*`, `repo_shadow/kai-llm-shim/Dockerfile`, etc.)
- docs -> architecture + dossier (`repo_shadow/ARCHITECTURE_SCHEMATIC.md`, `repo_shadow/system_azure_dossier.md`)

Note: this file is for planning only; do not treat as a completed move.
