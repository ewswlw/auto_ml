import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

def export_to_csv(data: pd.DataFrame, name: str, export_dir: str = None, overwrite: bool = True) -> str:
    """
    Export DataFrame to CSV file.
    
    Args:
        data (pd.DataFrame): Data to export
        name (str): Base name for the file
        export_dir (str, optional): Directory to export to. Defaults to None.
        overwrite (bool, optional): Whether to overwrite existing file. Defaults to True.
    
    Returns:
        str: Path to exported file
    """
    if export_dir is None:
        export_dir = os.path.join(os.getcwd(), 'data', 'exports')
    
    # Create directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)
    
    # Use simple filename without timestamp if overwriting
    if overwrite:
        filename = f"{name}.csv"
    else:
        # For non-overwrite, use timestamp to create unique name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{name}_{timestamp}.csv"
    
    filepath = os.path.join(export_dir, filename)
    
    try:
        # Use mode='w' to overwrite
        with open(filepath, 'w', newline='') as f:
            data.to_csv(f, index=True)
        logger.info(f"Successfully exported {name} to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error exporting {name} to CSV: {str(e)}")
        raise

def export_table_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Export a DataFrame to CSV.
    If the DataFrame has a DatetimeIndex, it will be preserved in the CSV.
    
    Args:
        df (pd.DataFrame): DataFrame to export
        output_path (Path): Path where to save the CSV file
    """
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Force overwrite by opening in write mode
    with open(output_path, 'w', newline='') as f:
        # Export to CSV with index and index label
        df.to_csv(f, index=True, index_label='Date')

def read_csv_to_df(file_path: str, fill: str = None, start_date_align: str = "no", overwrite: bool = True) -> pd.DataFrame:
    """
    Read CSV file into a DataFrame with date index and optional filling/alignment.
    
    Args:
        file_path (str): Path to CSV file
        fill (str, optional): Fill method for NaN values. Options: None, 'ffill', 'bfill', 'interpolate'
        start_date_align (str, optional): Whether to align start dates. Options: 'yes', 'no'
        overwrite (bool, optional): Whether to overwrite existing file when saving. Defaults to True.
        
    Returns:
        pd.DataFrame: DataFrame with date index
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Apply fill method if specified
        if fill == 'ffill':
            df = df.fillna(method='ffill')
        elif fill == 'bfill':
            df = df.fillna(method='bfill')
        elif fill == 'interpolate':
            df = df.interpolate()
            
        # Align start dates if requested
        if start_date_align.lower() == 'yes':
            # Find the latest start date
            latest_start = df.index.min()
            # Filter data to start from that date
            df = df[df.index >= latest_start]
            
        return df
        
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {str(e)}")
        raise
