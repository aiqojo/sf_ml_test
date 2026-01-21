"""Submit weather ML job using submit_directory()"""

import sys
from pathlib import Path

# Add src directory to Python path so imports work
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.snowflake.job_submit_utils import submit_directory_job

# Submit the job using the utility function
result = submit_directory_job(
    dir_path=str(src_dir),
    entrypoint="jobs/weather_ml_job.py",
    args=[
        "--data-table", "DWH_DEV.PSUPPLY.WEATHER_HISTORICAL",
        "--days-back", "30",
        "--limit", "10000",
        "--stage-name", "AI_ML.ML.STAGE_ML_SANDBOX_TEST",
    ],
    # Optional: add pip packages if needed
    # pip_requirements=["scikit-learn>=1.0.0"],
    # external_access_integrations=["PYPI_EAI"],
)

print(f"\n=== Job Summary ===")
print(f"Job ID: {result['job_id']}")
print(f"Status: {result['status']}")
print(f"Timed out: {result['timed_out']}")
if result['artifacts']:
    print(f"Artifacts downloaded: {len(result['artifacts'])} files")
