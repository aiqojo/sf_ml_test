"""Submit a warehouse benchmark job to Snowflake to test different warehouse sizes"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add src directory to Python path so imports work regardless of where script is run
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from snowflake.ml.jobs import remote
from snowflake.snowpark import Session
from utils.setup import (
    get_session_from_config,
    ensure_compute_pool_ready,
    ensure_stage_exists,
)
from utils.job_debug import (
    wait_for_job,
    show_job_logs,
    handle_job_result,
    diagnose_job_failure,
)
from utils.artifact_utils import download_job_artifacts

# Setup
session, session_params = get_session_from_config()

compute_pool = "ML_SANDBOX_TEST"
stage_name = "AI_ML.ML.STAGE_ML_SANDBOX_TEST"

ensure_stage_exists(session, stage_name)
ensure_compute_pool_ready(session, compute_pool)

# Define remote function
@remote(compute_pool, stage_name=stage_name, session=session, database="AI_ML", schema="ML")
def warehouse_benchmark():
    """
    Runs a suite of basic benchmark tests and times each operation.
    Tests various SQL operations to measure warehouse performance.
    """
    from snowflake.snowpark import Session
    from snowflake.snowpark.functions import (
        col, count, sum as sf_sum, avg, max as sf_max, min as sf_min,
        datediff, current_timestamp
    )
    from io import BytesIO
    import time
    import json
    
    # Get session from context
    session = Session.builder.getOrCreate()
    
    stage_name = "AI_ML.ML.STAGE_ML_SANDBOX_TEST"
    
    # Use weather_historical table - 8 billion rows, so we must be careful with limits
    # HARD LIMIT: Maximum rows to process in any single operation
    MAX_QUERY_ROWS = 10000000
    MAX_TABLE_ROWS = MAX_QUERY_ROWS * 10
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "warehouse": session.sql("SELECT CURRENT_WAREHOUSE()").collect()[0][0],
        "warehouse_size": session.sql("SELECT CURRENT_WAREHOUSE()").collect()[0][0],
        "max_query_rows": MAX_QUERY_ROWS,
        "max_table_rows": MAX_TABLE_ROWS,
        "tests": {}
    }
    # Get a tiny sample - limit immediately to avoid any large scans
    # NO DATE FILTERS - they cause massive scans even with limits
    # HARD LIMIT: Initial table limited to MAX_TABLE_ROWS, queries limited to MAX_QUERY_ROWS
    weather_df = session.table("DWH_DEV.PSUPPLY.WEATHER_HISTORICAL").limit(MAX_TABLE_ROWS)
    
    # Test 1: Simple scan with hard limit
    print("Test 1: Simple scan with hard limit...")
    start = time.time()
    # Just count the sampled data - no date filter to avoid scans
    limited_df = weather_df.limit(MAX_QUERY_ROWS)
    row_count = limited_df.count()
    elapsed = time.time() - start
    results["tests"]["limited_scan"] = {
        "description": f"Scan {MAX_QUERY_ROWS} rows (sampled)",
        "rows": row_count,
        "time_seconds": round(elapsed, 3)
    }
    print(f"  ✓ Scanned {row_count} rows in {elapsed:.3f}s")
    
    # Test 2: Filtered scan
    print("Test 2: Filtered scan...")
    start = time.time()
    # Simple filter on sampled data - no date filter to avoid scans
    filtered_df = weather_df.filter(col("VARIABLE").is_not_null()).limit(MAX_QUERY_ROWS)
    filtered_count = filtered_df.count()
    elapsed = time.time() - start
    results["tests"]["filtered_scan"] = {
        "description": f"Filter by VARIABLE, limited to {MAX_QUERY_ROWS} rows",
        "rows": filtered_count,
        "time_seconds": round(elapsed, 3)
    }
    print(f"  ✓ Filtered to {filtered_count} rows in {elapsed:.3f}s")
    
    # Test 3: Aggregation on limited subset
    print("Test 3: Aggregation (min/max/avg)...")
    start = time.time()
    # Use sampled data directly - already limited
    sample_for_agg = weather_df.limit(MAX_QUERY_ROWS)
    agg_df = sample_for_agg.agg(
        sf_min(col("LAT")).alias("min_lat"),
        sf_max(col("LAT")).alias("max_lat"),
        avg(col("LAT")).alias("avg_lat"),
        sf_min(col('"VALUE"')).alias("min_value"),
        sf_max(col('"VALUE"')).alias("max_value"),
        avg(col('"VALUE"')).alias("avg_value")
    )
    agg_result = agg_df.collect()[0]
    elapsed = time.time() - start
    results["tests"]["aggregation"] = {
        "description": f"Aggregation on {MAX_QUERY_ROWS} rows",
        "time_seconds": round(elapsed, 3),
        "result": {
            "min_lat": float(agg_result["MIN_LAT"]) if agg_result["MIN_LAT"] else None,
            "max_lat": float(agg_result["MAX_LAT"]) if agg_result["MAX_LAT"] else None,
            "avg_lat": float(agg_result["AVG_LAT"]) if agg_result["AVG_LAT"] else None,
            "min_value": float(agg_result["MIN_VALUE"]) if agg_result["MIN_VALUE"] else None,
            "max_value": float(agg_result["MAX_VALUE"]) if agg_result["MAX_VALUE"] else None,
            "avg_value": float(agg_result["AVG_VALUE"]) if agg_result["AVG_VALUE"] else None
        }
    }
    print(f"  ✓ Aggregation completed in {elapsed:.3f}s")
    
    # Test 4: Group by aggregation
    print("Test 4: Group by aggregation...")
    start = time.time()
    # Use sampled data directly
    sample_for_group = weather_df.limit(MAX_QUERY_ROWS)
    grouped_df = sample_for_group.select(
        (col("LAT").cast("int")).alias("lat_bucket"),
        (col("LON").cast("int")).alias("lon_bucket"),
        col("VARIABLE"),
        col('"VALUE"')  # VALUE is a reserved word, must be quoted
    ).group_by("lat_bucket", "lon_bucket", "VARIABLE").agg(
        count("*").alias("count"),
        avg(col('"VALUE"')).alias("avg_value")
    ).limit(MAX_QUERY_ROWS)  # Hard limit on groups
    group_count = grouped_df.count()
    elapsed = time.time() - start
    results["tests"]["group_by"] = {
        "description": f"Group by rounded lat/lon/variable on {MAX_QUERY_ROWS} rows (max {MAX_QUERY_ROWS} groups)",
        "groups": group_count,
        "time_seconds": round(elapsed, 3)
    }
    print(f"  ✓ Grouped into {group_count} groups in {elapsed:.3f}s")
    
    # Test 5: Join operation with both sides limited
    print("Test 5: Join operation...")
    try:
        grid_points_df = session.table("DWH_DEV.PSUPPLY.WEATHER_GRID_POINTS")
        # Hard limits on both sides - tiny datasets
        sample_grid = grid_points_df.limit(10)  # Only 10 grid points
        # Use sampled weather data - already limited
        sample_weather = weather_df.limit(MAX_QUERY_ROWS)
        
        start = time.time()
        # Join on lat/lon (exact match)
        joined = sample_grid.join(
            sample_weather,
            (sample_grid["LAT"] == sample_weather["LAT"]) & 
            (sample_grid["LON"] == sample_weather["LON"]),
            "inner"
        ).limit(MAX_QUERY_ROWS)  # Hard limit on join results
        join_count = joined.count()
        elapsed = time.time() - start
        results["tests"]["join"] = {
            "description": f"Join 10 grid points with {MAX_QUERY_ROWS} weather rows (max {MAX_QUERY_ROWS} results)",
            "result_rows": join_count,
            "time_seconds": round(elapsed, 3)
        }
        print(f"  ✓ Join produced {join_count} rows in {elapsed:.3f}s")
    except Exception as e:
        results["tests"]["join"] = {
            "description": "Join grid points with weather data",
            "error": str(e),
            "time_seconds": None
        }
        print(f"  ✗ Join test failed: {e}")
    
    # Test 6: Sorting with limit
    print("Test 6: Sorting...")
    start = time.time()
    # Use sampled data directly
    sample_for_sort = weather_df.limit(MAX_QUERY_ROWS)
    sorted_df = sample_for_sort.order_by(
        col("MSRMT_TIME").desc(), 
        col('"VALUE"').desc()
    ).limit(MAX_QUERY_ROWS)  # Hard limit
    sorted_count = sorted_df.count()
    elapsed = time.time() - start
    results["tests"]["sorting"] = {
        "description": f"Sort {MAX_QUERY_ROWS} rows by time/value desc, take top {MAX_QUERY_ROWS}",
        "rows": sorted_count,
        "time_seconds": round(elapsed, 3)
    }
    print(f"  ✓ Sorted and retrieved {sorted_count} rows in {elapsed:.3f}s")
    
    # Test 7: Window function with limit
    print("Test 7: Window function...")
    from snowflake.snowpark.window import Window
    start = time.time()
    sample_for_window = weather_df.limit(MAX_QUERY_ROWS)  # Use sampled data
    window_df = sample_for_window.select(
        col("MSRMT_TIME"),
        col("LAT"),
        col('"VALUE"'),
        avg(col('"VALUE"')).over(
            Window.partition_by((col("LAT").cast("int")))
        ).alias("avg_value_by_lat_bucket")
    ).limit(MAX_QUERY_ROWS)  # Hard limit
    window_count = window_df.count()
    elapsed = time.time() - start
    results["tests"]["window_function"] = {
        "description": f"Window function on {MAX_QUERY_ROWS} rows, partitioned by rounded LAT",
        "rows": window_count,
        "time_seconds": round(elapsed, 3)
    }
    print(f"  ✓ Window function completed on {window_count} rows in {elapsed:.3f}s")
    
    # Test 8: Complex query (multiple operations) with limits
    print("Test 8: Complex query (multiple operations)...")
    start = time.time()
    sample_for_complex = weather_df.limit(MAX_QUERY_ROWS)  # Use sampled data
    complex_df = (
        sample_for_complex.filter(col("LAT").is_not_null())
        .filter(col('"VALUE"').is_not_null())
        .select(
            (col("LAT").cast("int")).alias("lat_bucket"),
            (col("LON").cast("int")).alias("lon_bucket"),
            col("VARIABLE"),
            col('"VALUE"')
        )
        .group_by("lat_bucket", "lon_bucket", "VARIABLE")
        .agg(
            count("*").alias("count"),
            avg(col('"VALUE"')).alias("avg_value")
        )
        .filter(col("count") > 1)  # Lower threshold for small dataset
        .order_by(col("avg_value").desc())
        .limit(MAX_QUERY_ROWS)  # Hard limit
    )
    complex_count = complex_df.count()
    elapsed = time.time() - start
    results["tests"]["complex_query"] = {
        "description": f"Complex query: Filter -> Select -> Group -> Filter -> Sort -> Limit on {MAX_QUERY_ROWS} rows",
        "rows": complex_count,
        "time_seconds": round(elapsed, 3)
    }
    print(f"  ✓ Complex query returned {complex_count} rows in {elapsed:.3f}s")
    
    # Test 9: Data write (create temp table) with limit
    print("Test 9: Data write (temp table)...")
    try:
        temp_table_name = f"TEMP_BENCHMARK_{int(time.time())}"
        start = time.time()
        sample_for_write = weather_df.limit(MAX_QUERY_ROWS)  # Hard limit
        sample_for_write.write.mode("overwrite").save_as_table(temp_table_name, table_type="temporary")
        elapsed = time.time() - start
        # Verify write
        verify_count = session.table(temp_table_name).count()
        results["tests"]["data_write"] = {
            "description": f"Write {MAX_QUERY_ROWS} rows to temporary table",
            "rows_written": verify_count,
            "time_seconds": round(elapsed, 3),
            "temp_table": temp_table_name
        }
        print(f"  ✓ Wrote {verify_count} rows to temp table in {elapsed:.3f}s")
    except Exception as e:
        results["tests"]["data_write"] = {
            "description": "Write to temporary table",
            "error": str(e),
            "time_seconds": None
        }
        print(f"  ✗ Write test failed: {e}")
    
    # Test 10: Pandas conversion with limit
    print("Test 10: Pandas conversion...")
    start = time.time()
    sample_for_pandas = weather_df.limit(MAX_QUERY_ROWS)  # Hard limit
    pandas_df = sample_for_pandas.to_pandas()
    elapsed = time.time() - start
    results["tests"]["pandas_conversion"] = {
        "description": f"Convert {MAX_QUERY_ROWS} rows to pandas DataFrame",
        "rows": len(pandas_df),
        "time_seconds": round(elapsed, 3)
    }
    print(f"  ✓ Converted {len(pandas_df)} rows to pandas in {elapsed:.3f}s")
    
    # Calculate total time
    total_time = sum(
        test.get("time_seconds", 0) 
        for test in results["tests"].values() 
        if test.get("time_seconds") is not None
    )
    results["total_time_seconds"] = round(total_time, 3)
    
    # Save results to stage as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"{timestamp}_benchmark_results.json"
    json_data = json.dumps(results, indent=2)
    json_bytes = json_data.encode("utf-8")
    json_buffer = BytesIO(json_bytes)
    json_stage_path = f"@{stage_name}/output/{json_filename}"
    session.file.put_stream(
        json_buffer,
        json_stage_path,
        overwrite=True,
        auto_compress=False,
    )
    print(f"\n✓ Saved benchmark results to: {json_stage_path}")
    
    # Print summary
    print("\n=== Benchmark Summary ===")
    print(f"Total time: {total_time:.3f}s")
    print("\nIndividual test times:")
    for test_name, test_data in results["tests"].items():
        time_val = test_data.get("time_seconds")
        if time_val is not None:
            print(f"  {test_name}: {time_val:.3f}s")
        else:
            print(f"  {test_name}: FAILED")
    
    return {
        "timestamp": results["timestamp"],
        "warehouse": results["warehouse"],
        "max_query_rows": results["max_query_rows"],
        "max_table_rows": results["max_table_rows"],
        "total_time_seconds": results["total_time_seconds"],
        "tests": {k: {"time_seconds": v.get("time_seconds")} for k, v in results["tests"].items()},
        "json_stage_path": json_stage_path
    }

# Submit and monitor job
print("\n=== Submitting warehouse benchmark job ===")
try:
    job = warehouse_benchmark()
    
    final_status, timed_out, log_file = wait_for_job(job, timeout=3600)
    show_job_logs(job, log_file=log_file)
    result = handle_job_result(job, timed_out)
    
    # Download artifacts if job succeeded
    if result and "json_stage_path" in result:
        print("\n=== Downloading artifacts ===")
        artifacts_dir = Path(__file__).parent.parent.parent / "artifacts"
        downloaded = download_job_artifacts(
            session,
            result,
            artifacts_dir=artifacts_dir,
            stage_path_keys=["json_stage_path"]
        )
        for key, path in downloaded.items():
            print(f"✓ Downloaded {key}: {path}")
        
        # Print summary
        print("\n=== Benchmark Results Summary ===")
        print(f"Warehouse: {result.get('warehouse', 'N/A')}")
        print(f"Total time: {result.get('total_time_seconds', 'N/A')}s")
        print("\nTest breakdown:")
        for test_name, test_data in result.get("tests", {}).items():
            time_val = test_data.get("time_seconds")
            if time_val is not None:
                print(f"  {test_name}: {time_val:.3f}s")
            else:
                print(f"  {test_name}: FAILED")
except Exception as e:
    diagnose_job_failure(e, session, session_params)
