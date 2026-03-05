# FormWave

Monorepo orchestration for the **FormWave platform**.

This repository provides the **shared infrastructure and coordination layer** for the FormWave system.

Actual implementation lives in the submodules:

* `formwave-ai` – AI pipeline for data extraction, analysis, and model training
* `formwave-app` – backend API and frontend application

Each submodule contains its own documentation and startup instructions.

Refer to their README files for details.

---

# Repository Structure

```
formwave/
│
├── docker-compose.yml
│
├── formwave-ai/      # AI engine (submodule)
├── formwave-app/     # application (submodule)
│
└── infra/
    └── postgres/
        ├── migration/
        ├── backups/
        └── scripts/
```

This repository is responsible only for:

* shared infrastructure
* database lifecycle
* operational scripts

---

# Shared PostgreSQL

Both **FormWave AI** and **FormWave App** use the same PostgreSQL instance.

The database is started from this repository.

Start PostgreSQL:

```
docker compose up -d
```

Verify container:

```
docker ps
```

Expected container:

```
formwave-postgres
```

---

# Why a Shared Database

The system uses one PostgreSQL instance to keep all platform data in a single location.

Schemas are used to separate responsibilities.

Example layout:

```
core
ai_pipeline
app
```

* `core` – shared entities
* `ai_pipeline` – data extraction and AI processing
* `app` – application data

This approach allows:

* a single source of truth
* easier data exchange between AI and the application
* simpler infrastructure management

---

# Running the Platform

1. Clone repository with submodules

```
git clone --recurse-submodules <repo-url>
```

If already cloned:

```
git submodule update --init --recursive
```

2. Start infrastructure

```
docker compose up -d
```

3. Run components separately

For AI pipeline:

```
cd formwave-ai
```

Refer to:

```
formwave-ai/README.md
```

For application:

```
cd formwave-app
```

Refer to:

```
formwave-app/README.md
```

---

# Database Backups

Backup scripts are located in:

```
infra/postgres/scripts
```

Run backup:

```
./infra/postgres/scripts/backup_to_gdrive.sh
```

The dump will be created at:

```
infra/postgres/backups/formwave_latest.dump
```

After generation, upload the dump to the shared storage location (Google Drive).

---

# Notes

This repository should contain **only infrastructure and orchestration**.

Submodules contain:

* application code
* AI pipeline
* service-specific documentation

Refer to their README files for setup and usage.