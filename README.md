# FormWave

Monorepo for FormWave infrastructure and the AI/application subprojects.

This repository currently contains orchestration and infra assets plus the following local directories:

- `formwave-ai/` — AI pipeline sources, scripts, and model artifacts.
- `formwave-app/` — application code (backend/frontend).
- `infra/` — database migration scripts and backup helpers.

This README describes the repository's current state, known issues, and quick developer setup notes.

**Current status**

- **Infrastructure orchestration present:** `docker-compose.yml` is available to start PostgreSQL and MinIO.
- **AI pipeline code included:** `formwave-ai/` contains the pipeline and modules used for data processing and training.
- **Missing reproducible dependency manifests:** there is no `requirements.txt`, `pyproject.toml`, or similar at repository root or in submodules. Reproducible installs are not guaranteed.
- **No automated CI/tests:** there is no test suite or CI configuration in this repo.
- **Sensitive file committed:** a Google Cloud service account JSON is present in the tree: [formwave_ai/storage/fse570-oregon-formwave-2bfb03d8ed9a.json](formwave_ai/storage/fse570-oregon-formwave-2bfb03d8ed9a.json). Treat this as a security incident (see Security section).
- **Large backup in repo:** a DB dump exists under `infra/postgres/backups/` which may contain sensitive or large data.

**Security (action required)**

- Remove the committed service account from git history and revoke/rotate the key in GCP immediately.
- Do not run containers or scripts that rely on the checked-in JSON key. Use environment-injected credentials or CI secret stores instead.
- Remove large or sensitive backups from the repository and move them to secure storage. Consider `git filter-repo` or BFG to purge history.

Quick commands (examples):

```bash
# start infra (requires env vars set)
docker compose up -d

# inspect running containers
docker ps
```

Developer notes

- Before running anything that interacts with cloud APIs, configure credentials via environment variables or your local gcloud SDK.
- Add a dependency manifest in each Python component (e.g., `requirements.txt` or `pyproject.toml`) and pin versions.
- Replace ad-hoc `print()` debugging in the pipeline with structured `logging` and add a minimal test scaffold.

Recommended next steps

1. Rotate and remove the committed service account key (urgent).
2. Add `requirements.txt` / `pyproject.toml` for `formwave-ai` and `formwave-app` (short-term).
3. Add basic CI that installs deps, runs linters, and runs tests (short-term).
4. Remove large backups and stop committing generated data; add `.gitignore` entries (short-term).

Where to look

- Pipeline code: [formwave_ai/pipeline](formwave_ai/pipeline)
- Storage helpers: [formwave_ai/storage](formwave_ai/storage)
- Infra scripts and backups: [infra/postgres](infra/postgres)

If you want, I can: remove the secret from history, add `requirements.txt` for `formwave-ai`, and scaffold a basic GitHub Actions workflow. Tell me which to start with.