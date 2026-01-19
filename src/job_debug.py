"""Debug and logging utilities for Snowflake ML Jobs"""

import re
import time
import traceback
from pathlib import Path
from datetime import datetime
from snowflake.ml.jobs import get_job


def wait_for_job(job, timeout, poll_interval=15):
    """Wait for job completion with polling loop and overall timeout"""
    print(f"Waiting for job to complete (max {timeout}s, polling every {poll_interval}s)...")
    
    start_time = time.time()
    timed_out = False
    
    while True:
        elapsed = time.time() - start_time
        
        # Check overall timeout
        if elapsed >= timeout:
            timed_out = True
            print(f"\n⚠ Overall timeout ({timeout}s) reached")
            print(f"Current status: {job.status}")
            return job.status, timed_out
        
        # Check job status
        current_status = job.status
        
        # Print status every poll (overwrite same line, pad to clear previous)
        status_line = f"  [{int(elapsed)}s] Status: {current_status}"
        # Pad with spaces to ensure full overwrite (80 chars should be enough)
        padded_line = status_line.ljust(80)
        print(f"\r{padded_line}", end="", flush=True)
        
        # Terminal states - job is done
        if current_status in ["DONE", "FAILED", "CANCELLED"]:
            print(f"\nFinal status: {current_status} (completed in {int(elapsed)}s)")
            return current_status, timed_out
        
        # Non-terminal states - continue polling
        time.sleep(poll_interval)


def show_job_logs(job, tail_chars=10000):
    """Save job logs to file and show summary"""
    print(f"\n=== Job Logs ===")
    try:
        logs = job.get_logs(verbose=True) or ""
        if logs:
            # Create logs directory
            logs_dir = Path(__file__).parent.parent / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            # Generate filename from job ID and timestamp
            job_id = job.id.split(".")[-1] if "." in job.id else job.id
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = logs_dir / f"{job_id}_{timestamp}.log"
            
            # Write full logs to file
            with open(log_file, "w") as f:
                f.write(logs)
            
            # Show summary in console
            log_lines = logs.split("\n")
            print(f"✓ Logs saved to: {log_file}")
            print(f"  Total lines: {len(log_lines)}")
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
        print(f"  Required: GRANT CREATE SERVICE ON SCHEMA AI_ML.ML TO ROLE {session_params['role']};")
    
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
