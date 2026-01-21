"""Utilities for submitting directory-based Snowflake ML Jobs"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from snowflake.ml.jobs import submit_directory
from snowflake.snowpark import Session
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
from utils.snowflake.artifact_utils import download_job_artifacts
from utils.path_utils import get_repo_root


def submit_directory_job(
    dir_path: str,
    entrypoint: str,
    compute_pool: str = "ML_SANDBOX_TEST",
    stage_name: str = "AI_ML.ML.STAGE_ML_SANDBOX_TEST",
    args: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    external_access_integrations: Optional[List[str]] = None,
    session: Optional[Session] = None,
    session_params: Optional[Dict[str, str]] = None,
    timeout: int = 3600,
    download_artifacts: bool = True,
    artifacts_dir: Optional[Path] = None,
    artifact_keys: Optional[List[str]] = None,
    auto_setup: bool = True,
) -> Dict[str, Any]:
    """
    Submit a directory-based Snowflake ML Job with automatic setup, monitoring, and artifact download.
    
    Args:
        dir_path: Path to directory to upload (e.g., "src" or "/path/to/project")
        entrypoint: Entrypoint script relative to dir_path (e.g., "jobs/my_job.py")
        compute_pool: Compute pool name (default: "ML_SANDBOX_TEST")
        stage_name: Stage name for uploads (default: "AI_ML.ML.STAGE_ML_SANDBOX_TEST")
        args: CLI arguments to pass to entrypoint script (list of strings)
        pip_requirements: Optional pip packages to install (e.g., ["pandas>=2.0.0"])
        external_access_integrations: External access integrations for pip installs
        session: Snowpark session (if None, will create from config)
        session_params: Session parameters dict (if None, will load from config)
        timeout: Job timeout in seconds (default: 3600)
        download_artifacts: Whether to download artifacts from job result (default: True)
        artifacts_dir: Directory to download artifacts to (default: project_root/artifacts)
        artifact_keys: Keys in result dict that contain stage paths (default: looks in "outputs" dict)
        auto_setup: Whether to automatically ensure compute pool and stage exist (default: True)
        
    Returns:
        Dictionary with:
        - "job_id": Job ID string
        - "status": Final job status
        - "result": Job result dict (if successful)
        - "artifacts": Dict of downloaded artifact paths (if downloaded)
        - "timed_out": Whether job timed out
        
    Example:
        ```python
        result = submit_directory_job(
            dir_path="src",
            entrypoint="jobs/weather_ml_job.py",
            args=["--data-table", "DWH_DEV.PSUPPLY.WEATHER_HISTORICAL", "--days-back", "30"],
            pip_requirements=["scikit-learn>=1.0.0"],
        )
        print(f"Job {result['job_id']} completed with status {result['status']}")
        ```
    """
    # Setup session if not provided
    if session is None:
        session, session_params = get_session_from_config()
    elif session_params is None:
        # If session provided but not params, create empty dict
        session_params = {}
    
    # Auto-setup compute pool and stage
    if auto_setup:
        print(f"Ensuring compute pool '{compute_pool}' is ready...")
        ensure_compute_pool_ready(session, compute_pool)
        print(f"Ensuring stage '{stage_name}' exists...")
        ensure_stage_exists(session, stage_name)
    
    # Prepare args
    if args is None:
        args = []
    
    # Submit directory
    print(f"\n=== Submitting directory job ===")
    print(f"Directory: {dir_path}")
    print(f"Entrypoint: {entrypoint}")
    print(f"Compute pool: {compute_pool}")
    print(f"Stage: {stage_name}")
    if args:
        print(f"Args: {' '.join(args)}")
    if pip_requirements:
        print(f"Pip requirements: {', '.join(pip_requirements)}")
    
    try:
        job = submit_directory(
            dir_path=dir_path,
            compute_pool=compute_pool,
            entrypoint=entrypoint,
            stage_name=stage_name,
            args=args,
            pip_requirements=pip_requirements,
            external_access_integrations=external_access_integrations,
            session=session
        )
        
        print(f"Job submitted: {job.id}")
        
        # Wait for job to complete
        final_status, timed_out, log_file = wait_for_job(job, timeout=timeout)
        show_job_logs(job, log_file=log_file)
        result = handle_job_result(job, timed_out)
        
        # Download artifacts if requested and available
        artifacts = {}
        if download_artifacts and result:
            # Determine artifact keys
            if artifact_keys is None:
                # Default: look for "outputs" dict
                if "outputs" in result and result["outputs"]:
                    artifact_data = result["outputs"]
                    artifact_keys = list(artifact_data.keys())
                else:
                    # Fallback: look for common keys
                    artifact_keys = [k for k in result.keys() if k.endswith("_stage_path") or k.endswith("_path")]
                    artifact_data = {k: result[k] for k in artifact_keys if k in result}
            else:
                # Use provided keys
                artifact_data = {k: result.get(k) for k in artifact_keys if result.get(k)}
            
            if artifact_data:
                print("\n=== Downloading artifacts ===")
                if artifacts_dir is None:
                    # Default to artifacts directory in project root
                    artifacts_dir = get_repo_root() / "artifacts"
                
                downloaded = download_job_artifacts(
                    session,
                    artifact_data,
                    artifacts_dir=artifacts_dir,
                    stage_path_keys=list(artifact_data.keys())
                )
                artifacts = downloaded
                for key, path in downloaded.items():
                    print(f"✓ Downloaded {key}: {path}")
        
        return {
            "job_id": job.id,
            "status": final_status,
            "result": result,
            "artifacts": artifacts,
            "timed_out": timed_out,
        }
        
    except Exception as e:
        diagnose_job_failure(e, session, session_params)
        raise
