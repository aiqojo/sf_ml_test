"""Simple ML job using weather_historical table - entrypoint for submit_directory()"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, datediff, current_timestamp
import pandas as pd

from utils.snowflake.stage_utils import save_image_to_stage, save_dataframe_to_stage
from utils.plotting.plot_utils import create_correlation_heatmap, calculate_correlation_summary_stats


def load_weather_data(session: Session, table_name: str, days_back: int = 30, limit: int = 10000):
    """
    Load weather data from Snowflake table.
    
    Args:
        session: Snowpark session
        table_name: Name of weather table
        days_back: Number of days back to fetch data
        limit: Maximum number of records to fetch
        
    Returns:
        pandas DataFrame with weather data
    """
    print(f"Loading weather data from {table_name} (last {days_back} days, limit {limit})...")
    
    df = session.table(table_name).filter(
        datediff("day", col("MSRMT_TIME"), current_timestamp()) <= days_back
    ).limit(limit).to_pandas()
    
    print(f"Loaded {len(df)} records")
    return df


def prepare_features(df: pd.DataFrame):
    """
    Prepare features for ML - pivot weather variables into columns.
    
    Args:
        df: DataFrame with VARIABLE, VALUE, MSRMT_TIME columns
        
    Returns:
        DataFrame with variables as columns
    """
    print("Preparing features by pivoting weather variables...")
    
    weather_pivot = df.pivot_table(
        index='MSRMT_TIME',
        columns='VARIABLE',
        values='VALUE',
        aggfunc='mean'
    )
    
    weather_pivot = weather_pivot.dropna(axis=1, how='all').dropna(axis=0, how='all')
    
    print(f"Features prepared: {len(weather_pivot)} time points, {len(weather_pivot.columns)} variables")
    print(f"Variables: {list(weather_pivot.columns)}")
    
    return weather_pivot


def simple_baseline_ml(features_df: pd.DataFrame):
    """
    Simple baseline ML: calculate correlations and basic statistics.
    
    Args:
        features_df: DataFrame with weather variables as columns
        
    Returns:
        Dictionary with results
    """
    print("Running baseline ML analysis...")
    
    correlation_matrix = features_df.corr()
    
    summary_stats = calculate_correlation_summary_stats(correlation_matrix)
    
    feature_stats = features_df.describe().to_dict()
    
    correlations_flat = []
    for i, var1 in enumerate(correlation_matrix.columns):
        for j, var2 in enumerate(correlation_matrix.columns):
            if i != j:
                correlations_flat.append({
                    'variable1': var1,
                    'variable2': var2,
                    'correlation': correlation_matrix.loc[var1, var2]
                })
    
    correlations_df = pd.DataFrame(correlations_flat)
    top_correlations = correlations_df.nlargest(10, 'correlation')
    
    print(f"Top correlations found:")
    for _, row in top_correlations.iterrows():
        print(f"  {row['variable1']} <-> {row['variable2']}: {row['correlation']:.3f}")
    
    return {
        'correlation_matrix': correlation_matrix,
        'summary_stats': summary_stats,
        'feature_stats': feature_stats,
        'top_correlations': top_correlations
    }


def main(data_table: str, days_back: int = 30, limit: int = 10000, stage_name: str = None):
    """
    Main entrypoint for the ML job.
    
    Args:
        data_table: Name of weather table (e.g., "DWH_DEV.PSUPPLY.WEATHER_HISTORICAL")
        days_back: Number of days back to fetch data
        limit: Maximum number of records to fetch
        stage_name: Stage name for saving outputs (optional)
        
    Returns:
        Dictionary with job results
    """
    print("=" * 60)
    print("Weather ML Baseline Job")
    print("=" * 60)
    
    session = Session.builder.getOrCreate()
    
    weather_df = load_weather_data(session, data_table, days_back, limit)
    
    if weather_df.empty:
        return {"error": "No weather data found", "data_table": data_table}
    
    features_df = prepare_features(weather_df)
    
    if features_df.empty or len(features_df.columns) < 2:
        return {
            "error": "Insufficient data for ML analysis",
            "variables_found": list(features_df.columns) if not features_df.empty else []
        }
    
    results = simple_baseline_ml(features_df)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    outputs = {}
    if stage_name:
        print(f"\nSaving outputs to stage: {stage_name}")
        
        heatmap_buf = create_correlation_heatmap(
            results['correlation_matrix'],
            title=f"Weather Variable Correlations ({timestamp})"
        )
        png_path = save_image_to_stage(
            session,
            heatmap_buf,
            f"{timestamp}_weather_correlation.png",
            stage_name,
            subdirectory="output"
        )
        outputs['correlation_plot'] = png_path
        print(f"  Saved correlation plot: {png_path}")
        
        csv_path = save_dataframe_to_stage(
            session,
            results['correlation_matrix'],
            f"{timestamp}_correlation_matrix.csv",
            stage_name,
            subdirectory="output"
        )
        outputs['correlation_csv'] = csv_path
        print(f"  Saved correlation matrix: {csv_path}")
        
        top_corr_path = save_dataframe_to_stage(
            session,
            results['top_correlations'],
            f"{timestamp}_top_correlations.csv",
            stage_name,
            subdirectory="output"
        )
        outputs['top_correlations_csv'] = top_corr_path
        print(f"  Saved top correlations: {top_corr_path}")
    
    return {
        "status": "success",
        "timestamp": timestamp,
        "data_table": data_table,
        "records_loaded": len(weather_df),
        "time_points": len(features_df),
        "variables": list(features_df.columns),
        "summary_stats": results['summary_stats'],
        "outputs": outputs
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather ML Baseline Job")
    parser.add_argument("--data-table", required=True, help="Weather table name (e.g., DWH_DEV.PSUPPLY.WEATHER_HISTORICAL)")
    parser.add_argument("--days-back", type=int, default=30, help="Number of days back to fetch data")
    parser.add_argument("--limit", type=int, default=10000, help="Maximum number of records to fetch")
    parser.add_argument("--stage-name", help="Stage name for saving outputs (optional)")
    
    args = parser.parse_args()
    
    __return__ = main(
        data_table=args.data_table,
        days_back=args.days_back,
        limit=args.limit,
        stage_name=args.stage_name
    )
