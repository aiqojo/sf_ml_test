"""Utilities for creating plots and visualizations"""

from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt


def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: tuple = (12, 12),
    dpi: int = 300,
    cmap: str = "coolwarm",
    vmin: float = -1,
    vmax: float = 1,
) -> BytesIO:
    """
    Create a correlation heatmap plot and return as BytesIO buffer.
    
    Args:
        correlation_matrix: pandas DataFrame with correlation values
        title: Plot title
        figsize: Figure size tuple (width, height)
        dpi: Resolution for saved image
        cmap: Colormap name
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        
    Returns:
        BytesIO buffer containing PNG image data
    """
    buf = BytesIO()
    plt.figure(figsize=figsize)
    plt.imshow(correlation_matrix.values, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
    plt.colorbar(label='Correlation')
    plt.title(title)
    plt.xlabel('Variable')
    plt.ylabel('Variable')
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=dpi)
    plt.close()
    buf.seek(0)
    return buf


def calculate_correlation_summary_stats(correlation_matrix: pd.DataFrame) -> dict:
    """
    Calculate summary statistics for a correlation matrix.
    
    Args:
        correlation_matrix: pandas DataFrame with correlation values
        
    Returns:
        Dictionary with summary statistics:
        - max_correlation: Maximum correlation value
        - min_correlation: Minimum correlation value
        - mean_abs_correlation: Mean of absolute correlation values
    """
    return {
        "max_correlation": float(correlation_matrix.max().max()),
        "min_correlation": float(correlation_matrix.min().min()),
        "mean_abs_correlation": float(correlation_matrix.abs().mean().mean())
    }
