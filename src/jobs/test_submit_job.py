"""Submit a simple ML job to Snowflake"""

import sys
from pathlib import Path

# Add src directory to Python path so imports work regardless of where script is run
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from snowflake.ml.jobs import remote
from utils.snowflake.setup import (
    get_session_from_config,
    ensure_compute_pool_ready,
    ensure_stage_exists,
)
from utils.snowflake.job_debug import (
    wait_for_job,
    show_job_logs,
    handle_job_result,
    diagnose_job_failure,
)

# Setup
session, session_params = get_session_from_config()

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
    
    final_status, timed_out, log_file = wait_for_job(job, timeout=3600)
    show_job_logs(job, log_file=log_file)
    handle_job_result(job, timed_out)
except Exception as e:
    diagnose_job_failure(e, session, session_params)
