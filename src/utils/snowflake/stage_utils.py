"""Utilities for saving and downloading artifacts from Snowflake stages"""

from io import BytesIO
from pathlib import Path
from typing import Optional
from snowflake.snowpark import Session
import pandas as pd


def save_image_to_stage(
    session: Session,
    image_bytes: BytesIO,
    filename: str,
    stage_name: str,
    subdirectory: str = "output",
    overwrite: bool = True,
) -> str:
    """
    Save an image (PNG, JPEG, etc.) to a Snowflake stage.
    
    Args:
        session: Snowpark session
        image_bytes: BytesIO buffer containing image data
        filename: Name of the file (e.g., "plot.png")
        stage_name: Stage name (e.g., "AI_ML.ML.STAGE_ML_SANDBOX_TEST")
        subdirectory: Subdirectory within stage (default: "output")
        overwrite: Whether to overwrite existing files
        
    Returns:
        Stage path string (e.g., "@AI_ML.ML.STAGE_ML_SANDBOX_TEST/output/plot.png")
    """
    image_bytes.seek(0)
    
    stage_path = f"@{stage_name}/{subdirectory}/{filename}"
    
    session.file.put_stream(
        image_bytes,
        stage_path,
        overwrite=overwrite,
        auto_compress=False,
    )
    
    return stage_path


def save_csv_to_stage(
    session: Session,
    csv_data: str,
    filename: str,
    stage_name: str,
    subdirectory: str = "output",
    overwrite: bool = True,
) -> str:
    """
    Save CSV data (as string) to a Snowflake stage.
    
    Args:
        session: Snowpark session
        csv_data: CSV data as string
        filename: Name of the file (e.g., "data.csv")
        stage_name: Stage name (e.g., "AI_ML.ML.STAGE_ML_SANDBOX_TEST")
        subdirectory: Subdirectory within stage (default: "output")
        overwrite: Whether to overwrite existing files
        
    Returns:
        Stage path string
    """
    csv_bytes = csv_data.encode("utf-8")
    csv_buffer = BytesIO(csv_bytes)
    
    stage_path = f"@{stage_name}/{subdirectory}/{filename}"
    
    session.file.put_stream(
        csv_buffer,
        stage_path,
        overwrite=overwrite,
        auto_compress=False,
    )
    
    return stage_path


def save_dataframe_to_stage(
    session: Session,
    df: pd.DataFrame,
    filename: str,
    stage_name: str,
    subdirectory: str = "output",
    format: str = "csv",
    overwrite: bool = True,
) -> str:
    """
    Save a pandas DataFrame to a Snowflake stage as CSV or Parquet.
    
    Args:
        session: Snowpark session
        df: pandas DataFrame
        filename: Name of the file (e.g., "data.csv" or "data.parquet")
        stage_name: Stage name (e.g., "AI_ML.ML.STAGE_ML_SANDBOX_TEST")
        subdirectory: Subdirectory within stage (default: "output")
        format: File format ("csv" or "parquet")
        overwrite: Whether to overwrite existing files
        
    Returns:
        Stage path string
    """
    if format.lower() == "csv":
        csv_data = df.to_csv(index=True)
        return save_csv_to_stage(session, csv_data, filename, stage_name, subdirectory, overwrite)
    elif format.lower() == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for Parquet format")
        
        buffer = BytesIO()
        table = pa.Table.from_pandas(df)
        pq.write_table(table, buffer)
        buffer.seek(0)
        
        stage_path = f"@{stage_name}/{subdirectory}/{filename}"
        session.file.put_stream(
            buffer,
            stage_path,
            overwrite=overwrite,
            auto_compress=False,
        )
        return stage_path
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'parquet'")


def download_from_stage(
    session: Session,
    stage_path: str,
    local_path: Optional[str] = None,
    create_dirs: bool = True,
) -> Path:
    """
    Download a file from a Snowflake stage to local filesystem.
    
    Args:
        session: Snowpark session
        stage_path: Stage path (e.g., "@AI_ML.ML.STAGE_ML_SANDBOX_TEST/output/plot.png")
        local_path: Local directory or file path. If None, uses current directory.
                   If directory, extracts filename from stage_path.
        create_dirs: Whether to create parent directories if they don't exist
        
    Returns:
        Path object pointing to the downloaded file
    """
    if local_path is None:
        local_path = "."
    
    local_path_obj = Path(local_path)
    
    if local_path_obj.is_dir() or not local_path_obj.suffix:
        filename = stage_path.split("/")[-1]
        local_path_obj = local_path_obj / filename
    
    if create_dirs:
        local_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    session.file.get(stage_path, str(local_path_obj.parent))
    
    return local_path_obj


def download_from_stage_stream(
    session: Session,
    stage_path: str,
) -> bytes:
    """
    Download a file from a Snowflake stage as bytes (in-memory).
    
    Args:
        session: Snowpark session
        stage_path: Stage path (e.g., "@AI_ML.ML.STAGE_ML_SANDBOX_TEST/output/plot.png")
        
    Returns:
        File contents as bytes
    """
    with session.file.get_stream(stage_path) as f:
        return f.read()
