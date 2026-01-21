"""Utilities for downloading and managing job artifacts"""

from pathlib import Path
from typing import Dict, Optional, List
from snowflake.snowpark import Session
from utils.snowflake.stage_utils import download_from_stage
from utils.path_utils import get_repo_root


def download_job_artifacts(
    session: Session,
    result: Dict,
    artifacts_dir: Optional[Path] = None,
    stage_path_keys: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Download artifacts from a job result dictionary.
    
    Args:
        session: Snowpark session
        result: Job result dictionary containing stage_path keys
        artifacts_dir: Directory to download artifacts to (default: project_root/artifacts)
        stage_path_keys: List of keys in result dict that contain stage paths.
                        If None, looks for common keys: 'png_stage_path', 'csv_stage_path'
        
    Returns:
        Dictionary mapping stage_path_key -> local Path of downloaded file
    """
    if artifacts_dir is None:
        # Default to artifacts directory in project root
        # Assumes this is called from a job file in src/jobs/
        artifacts_dir = get_repo_root() / "artifacts"
    
    artifacts_dir.mkdir(exist_ok=True)
    
    if stage_path_keys is None:
        # Default keys to look for
        stage_path_keys = ["png_stage_path", "csv_stage_path"]
    
    downloaded = {}
    
    for key in stage_path_keys:
        if key in result and result[key]:
            stage_path = result[key]
            filename = stage_path.split("/")[-1]
            local_path = download_from_stage(
                session,
                stage_path,
                local_path=str(artifacts_dir / filename)
            )
            downloaded[key] = local_path
    
    return downloaded
