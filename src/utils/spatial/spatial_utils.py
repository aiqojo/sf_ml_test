"""Utilities for spatial/geographic data operations"""

from typing import List, Tuple, Any
from snowflake.snowpark.functions import col, abs as sf_abs
from snowflake.snowpark.column import Column
import pandas as pd


def build_multi_point_spatial_filter(
    lat_lon_pairs: List[Tuple[float, float]],
    lat_col: str = "LAT",
    lon_col: str = "LON",
    tolerance: float = 0.0001,
) -> Column:
    """
    Build a Snowpark filter condition for multiple lat/lon points using OR logic.
    
    Args:
        lat_lon_pairs: List of (lat, lon) tuples
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        tolerance: Tolerance in degrees for matching (default 0.0001 ≈ 11 meters)
        
    Returns:
        Combined filter condition (Column) that can be used with DataFrame.filter()
    """
    if not lat_lon_pairs:
        raise ValueError("lat_lon_pairs cannot be empty")
    
    filter_conditions = []
    for lat, lon in lat_lon_pairs:
        condition = (
            (sf_abs(col(lat_col) - lat) <= tolerance) & 
            (sf_abs(col(lon_col) - lon) <= tolerance)
        )
        filter_conditions.append(condition)
    
    # Combine all conditions with OR
    combined_filter = filter_conditions[0]
    for condition in filter_conditions[1:]:
        combined_filter = combined_filter | condition
    
    return combined_filter


def match_points_to_dataframe(
    df: pd.DataFrame,
    lat_lon_pairs: List[Tuple[float, float, Any]],
    lat_col: str = "LAT",
    lon_col: str = "LON",
    tolerance: float = 0.0001,
    match_col: str = "MATCH_ID",
) -> pd.DataFrame:
    """
    Match lat/lon points to a pandas DataFrame using tolerance-based matching.
    
    Args:
        df: pandas DataFrame with lat/lon columns
        lat_lon_pairs: List of tuples (lat, lon, match_id) where match_id is the value
                      to assign when matched
        lat_col: Name of latitude column in df
        lon_col: Name of longitude column in df
        tolerance: Tolerance in degrees for matching (default 0.0001 ≈ 11 meters)
        match_col: Name of column to add with match IDs
        
    Returns:
        DataFrame with match_col added (None for unmatched rows)
    """
    df = df.copy()
    df[match_col] = None
    
    for lat, lon, match_id in lat_lon_pairs:
        mask = (
            ((df[lat_col] - lat).abs() <= tolerance) &
            ((df[lon_col] - lon).abs() <= tolerance)
        )
        df.loc[mask, match_col] = match_id
    
    return df
