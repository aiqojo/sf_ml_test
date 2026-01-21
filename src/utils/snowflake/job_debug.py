"""Debug and logging utilities for Snowflake ML Jobs"""

import re
import time
import traceback
from pathlib import Path
from datetime import datetime
from snowflake.ml.jobs import get_job
from utils.path_utils import get_repo_root


def wait_for_job(job, timeout, poll_interval=15):
    """Wait for job completion with polling loop and overall timeout.
    Downloads logs during polling and at completion."""
    print(f"✓ Job created successfully! Job ID: {job.id}")
    print(f"Waiting for job to complete (timeout at {timeout}s, polling every {poll_interval}s)...")
    
    # Generate log filename at start (timestamp first for chronological sorting)
    logs_dir = get_repo_root() / "logs"
    logs_dir.mkdir(exist_ok=True)
    job_id = job.id.split(".")[-1] if "." in job.id else job.id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{timestamp}_{job_id}.log"
    
    start_time = time.time()
    timed_out = False
    
    while True:
        elapsed = time.time() - start_time
        
        # Check overall timeout
        if elapsed >= timeout:
            timed_out = True
            print(f"\n⚠ Overall timeout ({timeout}s) reached")
            print(f"Current status: {job.status}")
            # Download final logs before returning
            _download_logs(job, log_file)
            return job.status, timed_out, log_file
        
        # Check job status
        current_status = job.status
        
        # Download logs during polling (overwrites previous)
        log_size = _download_logs(job, log_file)
        
        # Shorten log filename for display (just show job ID part)
        log_display = log_file.stem  # filename without .log extension
        if "_" in log_display:
            # Extract just the job ID part (last part after all underscores)
            parts = log_display.split("_")
            if len(parts) >= 2:
                log_display = parts[-1]  # Just the job ID (last part)
        log_info = f"{log_display}.log" if log_size > 0 else f"{log_display}..."
        
        # print status every poll (overwrite same line, pad to clear previous)
        status_line = f"  [{int(elapsed)}s] {current_status} | log: {log_info}"
        # Pad with spaces to ensure full overwrite (120 chars to handle long lines)
        padded_line = status_line.ljust(120)
        print(f"\r{padded_line}", end="", flush=True)
        
        # Terminal states - job is done
        if current_status in ["DONE", "FAILED", "CANCELLED"]:
            print(f"\nFinal status: {current_status} (completed in {int(elapsed)}s)")
            # Download final logs
            _download_logs(job, log_file)
            return current_status, timed_out, log_file
        
        # Non-terminal states - continue polling
        time.sleep(poll_interval)


def _download_logs(job, log_file):
    """Helper to download job logs to file (overwrites existing).
    Returns the size of logs written (0 if none)."""
    try:
        logs = job.get_logs(verbose=True) or ""
        if logs:
            with open(log_file, "w") as f:
                f.write(logs)
            return len(logs)
        return 0
    except Exception as e:
        # Log first failure, but don't spam - only print once per unique error
        # This helps debug if logs aren't downloading
        if not hasattr(_download_logs, '_last_error') or _download_logs._last_error != str(e):
            _download_logs._last_error = str(e)
            # Only print if it's not a "no logs yet" type error
            if "not found" not in str(e).lower() and "not available" not in str(e).lower():
                print(f"\n⚠ Could not download logs during polling: {e}")
        return 0


def show_job_logs(job, tail_chars=10000, log_file=None):
    """Save job logs to file and show summary.
    If log_file is provided, will download fresh logs to that file.
    Otherwise, generates a new filename."""
    print(f"\n=== Job Logs ===")
    try:
        logs = job.get_logs(verbose=True) or ""
        if logs:
            # Use provided log_file or generate new one (timestamp first for chronological sorting)
            if log_file is None:
                logs_dir = get_repo_root() / "logs"
                logs_dir.mkdir(exist_ok=True)
                job_id = job.id.split(".")[-1] if "." in job.id else job.id
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = logs_dir / f"{timestamp}_{job_id}.log"
            
            # Write full logs to file (overwrites if exists)
            with open(log_file, "w") as f:
                f.write(logs)
            
            # Show summary in console
            log_lines = logs.split("\n")
            print(f"✓ Logs saved to: {log_file}")
            print(f"  Total lines: {len(log_lines)}")
            
            # Show tail of logs if requested
            if tail_chars > 0 and len(logs) > tail_chars:
                print(f"\n=== Last {tail_chars} characters of logs ===")
                print(logs[-tail_chars:])
        else:
            print("No logs available")
    except Exception as log_err:
        print(f"✗ Could not get logs: {log_err}")
        print(f"Traceback: {traceback.format_exc()}")


def handle_job_result(job, timed_out=False):
    """Handle job result based on status"""
    if job.status == "DONE":
        print(f"\n=== Job Result ===")
        try:
            result = job.result()
            print(f"Result: {result}")
            return result
        except Exception as result_err:
            print(f"✗ Could not get result: {result_err}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
    elif job.status == "FAILED":
        print(f"\n✗ Job failed - check logs above for details")
        return None
    else:
        # Timeout or still running - not a failure, just incomplete
        print(f"\n⚠ Job status: {job.status}")
        if timed_out:
            print(f"  Job will continue running - check Snowflake UI for final status")
        else:
            print(f"  Check Snowflake UI for final status")
        return None


def diagnose_job_failure(error, session, session_params):
    """Diagnose job creation failure"""
    error_msg = str(error)
    job_id_match = re.search(r'HELLO_[\w]+', error_msg)
    job_id = job_id_match.group(0) if job_id_match else None
    
    print(f"\n✗ Job creation failed: {error_msg[:200]}...")
    
    # Check if it's a permission error
    if "insufficient privileges" in error_msg.lower() or "access control" in error_msg.lower():
        print(f"\n  ⚠ Permission error detected!")
        role = session_params.get("role") if session_params else None
        role_display = role if role else "<role>"
        print(f"  Required: GRANT CREATE SERVICE ON SCHEMA AI_ML.ML TO ROLE {role_display};")
    
    if job_id:
        print(f"\n  Job ID: {job_id}")
        
        # Try to get job status using correct API
        try:
            job_obj = get_job(job_id, session=session)
            print(f"  ✓ Job exists! Status: {job_obj.status}")
            print(f"\n  === Container Logs (tail) ===")
            logs = job_obj.get_logs(verbose=True) or ""
            if logs:
                print(logs[-4000:])
            else:
                print("  No logs available yet")
        except Exception as job_err:
            print(f"  ✗ Failed to get job via API: {job_err}")
            print(f"  Traceback: {traceback.format_exc()}")
            
            # Try SQL-based job history query
            try:
                job_history = session.sql(f"""
                    SELECT name, status, message, created_time, completed_time
                    FROM TABLE(SNOWFLAKE.SPCS.GET_JOB_HISTORY(
                        created_time_start => DATEADD('hour', -6, CURRENT_TIMESTAMP()),
                        result_limit => 100
                    ))
                    WHERE name = '{job_id}'
                    ORDER BY created_time DESC
                    LIMIT 1
                """).collect()
                
                if job_history:
                    row = job_history[0]
                    status = row["status"] if "status" in row else "UNKNOWN"
                    print(f"  Status: {status}")
                    if "message" in row and row["message"]:
                        print(f"  Message: {row['message']}")
                else:
                    print(f"  Job not found in history yet")
            except Exception as hist_err:
                print(f"  ✗ Could not query job history: {hist_err}")
                print(f"  Traceback: {traceback.format_exc()}")
            
            # Try container logs
            try:
                logs_query = f"""
                SELECT *
                FROM TABLE(AI_ML.ML.{job_id}!SPCS_GET_LOGS(
                    START_TIME => DATEADD('hour', -6, CURRENT_TIMESTAMP())
                ))
                ORDER BY timestamp DESC
                LIMIT 50
                """
                container_logs = session.sql(logs_query).collect()
                if container_logs:
                    print(f"\n  === Container Logs (last 50 lines) ===")
                    for log_row in container_logs[:50]:
                        log_msg = log_row["LOG"] if "LOG" in log_row else str(log_row)
                        timestamp = log_row["TIMESTAMP"] if "TIMESTAMP" in log_row else ""
                        print(f"  [{timestamp}] {log_msg}")
            except Exception as log_err:
                print(f"  ✗ Could not get container logs: {log_err}")
                if "insufficient privileges" in str(log_err).lower():
                    print(f"  Need MONITOR privilege to read logs")
                else:
                    print(f"  Traceback: {traceback.format_exc()}")
        
        print(f"\n  Check Snowflake UI: Compute → Jobs → {job_id}")
