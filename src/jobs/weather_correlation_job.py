"""Submit a weather correlation analysis job to Snowflake"""

import sys
from pathlib import Path

# Add src directory to Python path so imports work regardless of where script is run
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from snowflake.ml.jobs import remote
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

# Setup
session, session_params = get_session_from_config()

compute_pool = "ML_SANDBOX_TEST"
stage_name = "AI_ML.ML.STAGE_ML_SANDBOX_TEST"

ensure_stage_exists(session, stage_name)
ensure_compute_pool_ready(session, compute_pool)

# Define remote function
@remote(compute_pool, stage_name=stage_name, session=session, database="AI_ML", schema="ML")
def weather_correlation_analysis():
    """
    Randomly selects a grid point and analyzes weather variable correlations
    for the most recent week of data.
    """
    from snowflake.snowpark import Session
    from snowflake.snowpark.functions import col, datediff, current_timestamp, random, abs as sf_abs
    from io import BytesIO
    from datetime import datetime
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Get session from context (Snowflake ML jobs provide this)
    session = Session.builder.getOrCreate()
    
    # Stage name (should match the one in @remote decorator)
    stage_name = "AI_ML.ML.STAGE_ML_SANDBOX_TEST"
    
    # Step 1: Randomly select 10 grid points
    print("Selecting 10 random grid points...")
    grid_points_df = session.table("DWH_DEV.PSUPPLY.WEATHER_GRID_POINTS").order_by(random()).limit(10).to_pandas()
    
    if grid_points_df.empty:
        return {"error": "No grid points found"}
    
    print(f"Selected {len(grid_points_df)} grid points")
    
    # Generate timestamp for filenames (shared across all files in this run)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use tolerance-based matching for lat/lon (0.0001 degrees ≈ 11 meters)
    tolerance = 0.0001
    
    # Step 2: Fetch all weather data for all grid points in a single query
    print("\nFetching weather data for all grid points...")
    
    # Create filter conditions for all grid points using OR
    filter_conditions = []
    grid_info = []
    
    for idx, row in grid_points_df.iterrows():
        grid_id = int(row['GRID_ID'])
        lat = float(row['LAT'])
        lon = float(row['LON'])
        grid_info.append({"grid_id": grid_id, "lat": lat, "lon": lon})
        
        # Create tolerance-based condition for this grid point
        condition = (
            (sf_abs(col("LAT") - lat) <= tolerance) & 
            (sf_abs(col("LON") - lon) <= tolerance)
        )
        filter_conditions.append(condition)
    
    # Combine all conditions with OR
    combined_filter = filter_conditions[0]
    for condition in filter_conditions[1:]:
        combined_filter = combined_filter | condition
    
    # Add date filter and fetch all data at once
    weather_df = session.table("DWH_DEV.PSUPPLY.WEATHER_HISTORICAL").filter(
        combined_filter &
        (datediff("day", col("MSRMT_TIME"), current_timestamp()) <= 60)
    ).to_pandas()
    
    print(f"Retrieved {len(weather_df)} total weather records")
    
    # Step 3: Add grid_id to weather data by matching lat/lon
    print("\nMatching weather data to grid points...")
    weather_df['GRID_ID'] = None
    
    for grid_point in grid_info:
        grid_id = grid_point["grid_id"]
        lat = grid_point["lat"]
        lon = grid_point["lon"]
        
        # Match weather data to this grid point
        mask = (
            ((weather_df['LAT'] - lat).abs() <= tolerance) &
            ((weather_df['LON'] - lon).abs() <= tolerance)
        )
        weather_df.loc[mask, 'GRID_ID'] = grid_id
    
    # Filter out rows that didn't match any grid point
    weather_df = weather_df[weather_df['GRID_ID'].notna()].copy()
    
    if weather_df.empty:
        return {
            "error": "No weather data found for any of the selected grid points",
            "grid_points": grid_info
        }
    
    # Count records per grid point
    grid_counts = weather_df['GRID_ID'].value_counts().to_dict()
    print(f"Records per grid point:")
    for grid_id, count in grid_counts.items():
        print(f"  Grid {grid_id}: {count} records")
    
    # Step 4: Pivot data to have variables as columns (across all grid points)
    print("\nPreparing data for correlation analysis (across all grid points)...")
    
    # Pivot: MSRMT_TIME as index, VARIABLE as columns, VALUE as values
    # This combines data from all grid points
    weather_pivot = weather_df.pivot_table(
        index='MSRMT_TIME',
        columns='VARIABLE',
        values='VALUE',
        aggfunc='mean'  # If there are multiple values for same time/variable, take mean
    )
    
    # Remove any columns/rows with all NaN
    weather_pivot = weather_pivot.dropna(axis=1, how='all').dropna(axis=0, how='all')
    
    if weather_pivot.empty or len(weather_pivot.columns) < 2:
        return {
            "error": "Insufficient data for correlation analysis (need at least 2 variables)",
            "variables_found": list(weather_pivot.columns) if not weather_pivot.empty else [],
            "grid_points": grid_info
        }
    
    print(f"Variables found: {list(weather_pivot.columns)}")
    print(f"Time points: {len(weather_pivot)}")
    
    # Step 5: Calculate correlation matrix across all variables
    print("Calculating correlation matrix...")
    correlation_matrix = weather_pivot.corr()
    
    # Step 6: Create correlation plot and save to stage
    print("Creating correlation plot...")
    
    # Create heatmap image
    buf = BytesIO()
    plt.figure(figsize=(12, 12))
    plt.imshow(correlation_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
    plt.colorbar(label='Correlation')
    plt.title(f'Weather Variable Correlations - Across {len(grid_info)} Grid Points')
    plt.xlabel('Variable')
    plt.ylabel('Variable')
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    buf.seek(0)
    
    # Save image to stage
    grid_ids_str = "_".join([str(g["grid_id"]) for g in grid_info])
    png_filename = f"{timestamp}_corr_grids_{grid_ids_str}.png"
    png_stage_path = f"@{stage_name}/output/{png_filename}"
    session.file.put_stream(
        buf,
        png_stage_path,
        overwrite=True,
        auto_compress=False,
    )
    print(f"Saved plot to: {png_stage_path}")
    
    # Save correlation matrix CSV to stage
    csv_filename = f"{timestamp}_corr_matrix_grids_{grid_ids_str}.csv"
    csv_data = correlation_matrix.to_csv()
    csv_bytes = csv_data.encode("utf-8")
    csv_buffer = BytesIO(csv_bytes)
    csv_stage_path = f"@{stage_name}/output/{csv_filename}"
    session.file.put_stream(
        csv_buffer,
        csv_stage_path,
        overwrite=True,
        auto_compress=False,
    )
    print(f"Saved CSV to: {csv_stage_path}")
    
    # Return summary
    return {
        "timestamp": timestamp,
        "grid_points": grid_info,
        "grid_counts": grid_counts,
        "variables": list(correlation_matrix.columns),
        "num_time_points": len(weather_pivot),
        "png_stage_path": png_stage_path,
        "csv_stage_path": csv_stage_path,
        "summary_stats": {
            "max_correlation": float(correlation_matrix.max().max()),
            "min_correlation": float(correlation_matrix.min().min()),
            "mean_abs_correlation": float(correlation_matrix.abs().mean().mean())
        }
    }

# Submit and monitor job
print("\n=== Submitting weather correlation job ===")
try:
    job = weather_correlation_analysis()
    
    final_status, timed_out, log_file = wait_for_job(job, timeout=3600)
    show_job_logs(job, log_file=log_file)
    result = handle_job_result(job, timed_out)
    
    # Download artifacts if job succeeded
    if result and "png_stage_path" in result:
        print("\n=== Downloading artifacts ===")
        artifacts_dir = Path(__file__).parent.parent.parent / "artifacts"
        downloaded = download_job_artifacts(
            session,
            result,
            artifacts_dir=artifacts_dir,
            stage_path_keys=["png_stage_path", "csv_stage_path"]
        )
        for key, path in downloaded.items():
            print(f"✓ Downloaded {key}: {path}")
except Exception as e:
    diagnose_job_failure(e, session, session_params)
