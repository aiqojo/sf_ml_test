# Snowflake ML Jobs

Utilities for submitting and monitoring Snowflake ML Jobs.

## Quick Start

### Using `submit_directory()` (Recommended)

**1. Create job entrypoint** (`src/jobs/my_job.py`):
```python
import argparse
from snowflake.snowpark import Session

def main(data_table: str):
    session = Session.builder.getOrCreate()
    df = session.table(data_table).to_pandas()
    # Your ML code here
    return {"status": "success"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-table", required=True)
    args = parser.parse_args()
    __return__ = main(args.data_table)
```

**2. Create submission script** (`src/jobs/my_submit.py`):
```python
from utils.job_submit_utils import submit_directory_job
from pathlib import Path

src_dir = Path(__file__).parent.parent

result = submit_directory_job(
    dir_path=str(src_dir),
    entrypoint="jobs/my_job.py",
    args=["--data-table", "MY_SCHEMA.MY_TABLE"],
    # pip_requirements=["pandas>=2.0.0"],
)
```

### Using `@remote` decorator (Legacy)

See `src/jobs/test_submit_job.py` for example.

## Project Structure

- `src/jobs/` - Job scripts
  - `*_job.py` - Job entrypoints (run remotely)
  - `*_submit.py` - Submission scripts (run locally)
- `src/utils/` - Utility modules
- `logs/` - Job log files (gitignored)
- `artifacts/` - Downloaded job outputs (gitignored)

## Utilities

### Job Submission (`utils/job_submit_utils.py`)

- **`submit_directory_job(...)`** - Submit directory-based jobs with automatic setup, monitoring, and artifact download

### Setup (`utils/setup.py`)

- **`get_session_from_config(config_path=None)`** - Loads Snowflake session from `.snowflake/config.toml`
- **`ensure_compute_pool_ready(session, target_pool, max_wait=60)`** - Ensures compute pool is ready
- **`ensure_stage_exists(session, stage_name)`** - Creates stage if needed

### Debug/Logging (`utils/job_debug.py`)

- **`wait_for_job(job, timeout=300)`** - Waits for job completion
- **`show_job_logs(job, tail_chars=6000)`** - Shows container logs
- **`handle_job_result(job, timed_out=False)`** - Processes job results
- **`diagnose_job_failure(error, session, session_params)`** - Failure diagnostics

## Naming Convention

- **`*_job.py`** - Job entrypoints executed remotely (use `argparse`, return via `__return__`)
- **`*_submit.py`** - Submission scripts run locally (call `submit_directory_job()`)

## Examples

- `src/jobs/weather_ml_job.py` + `src/jobs/weather_ml_submit.py` - Directory-based job
- `src/jobs/test_submit_job.py` - Legacy `@remote` decorator example
