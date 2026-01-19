"""Setup utilities for Snowflake ML Jobs"""

from snowflake.snowpark import Session
import tomli
import time
from pathlib import Path


def get_session_from_config(config_path=None):
    """Load Snowflake session from config.toml"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / ".snowflake" / "config.toml"
    
    with open(config_path, "rb") as f:
        config = tomli.load(f)
        connection_params = config["connections"]["ML_connection"]
    
    session_params = {
        "account": connection_params["SNOWFLAKE_ACCOUNT"],
        "user": connection_params["SNOWFLAKE_USER"],
        "role": connection_params["SNOWFLAKE_ROLE"],
        "warehouse": connection_params["SNOWFLAKE_WAREHOUSE"],
        "database": connection_params["SNOWFLAKE_DATABASE"],
        "schema": connection_params["SNOWFLAKE_SCHEMA"],
        "authenticator": connection_params["SF_CONNECTION_TYPE"].lower(),
    }
    
    session = Session.builder.configs(session_params).getOrCreate()
    return session, session_params


def ensure_compute_pool_ready(session, target_pool, max_wait=60):
    """Ensure compute pool is in IDLE or RUNNING state"""
    pools = session.sql("SHOW COMPUTE POOLS").collect()
    for pool in pools:
        if pool["name"].upper() == target_pool.upper():
            state = pool["state"]
            if state in ['IDLE', 'RUNNING']:
                return True
            elif state == 'SUSPENDED':
                print(f"Resuming compute pool {target_pool}...")
                try:
                    session.sql(f"ALTER COMPUTE POOL {target_pool} RESUME").collect()
                    wait_time = 0
                    while wait_time < max_wait:
                        time.sleep(3)
                        wait_time += 3
                        pools = session.sql("SHOW COMPUTE POOLS").collect()
                        for p in pools:
                            if p["name"].upper() == target_pool.upper():
                                current_state = p["state"]
                                if current_state in ['IDLE', 'RUNNING']:
                                    return True
                                elif current_state == 'SUSPENDED':
                                    print(f"✗ Compute pool failed to start")
                                    return False
                    print(f"⚠ Compute pool still starting after {max_wait}s")
                    return False
                except Exception as e:
                    print(f"✗ Failed to resume compute pool: {e}")
                    return False
            elif state == 'STARTING':
                print(f"Waiting for compute pool to become ready...")
                wait_time = 0
                while wait_time < max_wait:
                    time.sleep(3)
                    wait_time += 3
                    pools = session.sql("SHOW COMPUTE POOLS").collect()
                    for p in pools:
                        if p["name"].upper() == target_pool.upper():
                            if p["state"] in ['IDLE', 'RUNNING']:
                                return True
                print(f"⚠ Compute pool still starting after {max_wait}s")
                return False
            else:
                print(f"⚠ Compute pool state is {state} - may cause issues")
                return False
    print(f"✗ Compute pool {target_pool} not found")
    return False


def ensure_stage_exists(session, stage_name):
    """Ensure stage exists"""
    session.sql(f"CREATE STAGE IF NOT EXISTS {stage_name}").collect()
    print(f"✓ Stage ready: {stage_name}")
