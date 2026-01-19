"""Submit a simple ML job to Snowflake"""

from snowflake.ml.jobs import remote
from setup import (
    get_session_from_config,
    ensure_compute_pool_ready,
    ensure_stage_exists,
)
from job_debug import (
    wait_for_job,
    show_job_logs,
    handle_job_result,
    diagnose_job_failure,
)

# Setup
session, session_params = get_session_from_config()
print(f"✓ Connected to Snowflake")
print(f"  Version: {session.sql('select current_version()').collect()[0][0]}")

compute_pool = "ML_SANDBOX_TEST"
stage_name = "AI_ML.ML.STAGE_ML_SANDBOX_TEST"

ensure_stage_exists(session, stage_name)
ensure_compute_pool_ready(session, compute_pool)

# Define remote function (keep it simple - no external dependencies)
@remote(compute_pool, stage_name=stage_name, session=session, database="AI_ML", schema="ML")
def hello():
    return "Hello from remote job!"

# Submit and monitor job
print("\n=== Submitting job ===")
try:
    job = hello()
    print(f"✓ Job created successfully! Job ID: {job.id}")
    
    final_status, timed_out = wait_for_job(job, timeout=3600)
    show_job_logs(job)
    handle_job_result(job, timed_out)
except Exception as e:
    diagnose_job_failure(e, session, session_params)