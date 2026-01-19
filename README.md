# Snowflake ML Jobs

Utilities for submitting and monitoring Snowflake ML Jobs.

## Quick Start

### 1. Setup

```python
from utils.setup import get_session_from_config, ensure_compute_pool_ready, ensure_stage_exists
from snowflake.ml.jobs import remote

session, session_params = get_session_from_config()
ensure_stage_exists(session, "AI_ML.ML.STAGE_ML_SANDBOX_TEST")
ensure_compute_pool_ready(session, "ML_SANDBOX_TEST")
```

### 2. Define Remote Function

```python
@remote(
    "ML_SANDBOX_TEST",
    stage_name="AI_ML.ML.STAGE_ML_SANDBOX_TEST",
    session=session,
    database="AI_ML",
    schema="ML"
)
def my_job():
    # Keep function simple - avoid external module dependencies
    return "Hello from remote job!"
```

### 3. Submit and Monitor

```python
from utils.job_debug import wait_for_job, show_job_logs, handle_job_result, diagnose_job_failure

try:
    job = my_job()
    final_status, timed_out = wait_for_job(job, timeout=300)
    show_job_logs(job)
    handle_job_result(job, timed_out)
except Exception as e:
    diagnose_job_failure(e, session, session_params)
```

## Project Structure

- `src/jobs/` - Job submission scripts
- `src/utils/` - Utility modules (setup, debugging, logging)
- `logs/` - Job log files (gitignored)

## Utilities

### Setup (`utils/setup.py`)

- **`get_session_from_config(config_path=None)`** - Loads Snowflake session from `.snowflake/config.toml`
- **`ensure_compute_pool_ready(session, target_pool, max_wait=60)`** - Ensures compute pool is `IDLE` or `RUNNING` (auto-resumes if `SUSPENDED`)
- **`ensure_stage_exists(session, stage_name)`** - Creates stage if it doesn't exist

### Debug/Logging (`utils/job_debug.py`)

- **`wait_for_job(job, timeout=300)`** - Waits for job completion, handles timeouts gracefully
- **`show_job_logs(job, tail_chars=6000)`** - Prints tail of container logs
- **`handle_job_result(job, timed_out=False)`** - Processes result based on status (`DONE`, `FAILED`, etc.)
- **`diagnose_job_failure(error, session, session_params)`** - Detailed failure diagnostics (permissions, job history, container logs)

## Notes

- **Cold starts**: First job can take 2-5 minutes (Ray runtime boot). Use `timeout=300` minimum.
- **Function isolation**: Keep remote functions simple. Avoid importing modules that aren't available in the container.
- **Timeouts**: Timeouts are non-fatal - jobs continue running. Check Snowflake UI for final status.
- **Permissions**: Requires `CREATE SERVICE` on schema, `USAGE` on compute pool, `WRITE` on stage, `MONITOR` for logs.

## Example

See `src/jobs/test_submit_job.py` for a complete example.
