"""Debug and logging utilities for Snowflake ML Jobs"""

import re
from snowflake.ml.jobs import get_job


def wait_for_job(job, timeout=300):
    """Wait for job completion with timeout handling"""
    print(f"Waiting for job to complete (max {timeout}s)...")
    timed_out = False
    try:
        final_status = job.wait(timeout=timeout)
        print(f"Final status: {final_status}")
        return final_status, timed_out
    except Exception as timeout_err:
        error_str = str(timeout_err)
        if "did not complete within" in error_str or "timeout" in error_str.lower():
            timed_out = True
            print(f"⚠ Job still running after {timeout}s timeout")
            print(f"Current status: {job.status}")
            return job.status, timed_out
        else:
            raise


def show_job_logs(job, tail_chars=6000):
    """Show tail of job logs"""
    print(f"\n=== Job Logs (tail) ===")
    try:
        logs = job.get_logs(verbose=True) or ""
        if logs:
            print(logs[-tail_chars:])  # Last N chars
        else:
            print("No logs available")
    except Exception as log_err:
        print(f"Could not get logs: {log_err}")


def handle_job_result(job, timed_out=False):
    """Handle job result based on status"""
    if job.status == "DONE":
        print(f"\n=== Job Result ===")
        try:
            result = job.result()
            print(f"Result: {result}")
            return result
        except Exception as result_err:
            print(f"Could not get result: {result_err}")
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
                print(f"  Could not query job history: {hist_err}")
            
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
                if "insufficient privileges" in str(log_err).lower():
                    print(f"  Need MONITOR privilege to read logs")
                else:
                    print(f"  Could not get container logs: {log_err}")
        
        print(f"\n  Check Snowflake UI: Compute → Jobs → {job_id}")
