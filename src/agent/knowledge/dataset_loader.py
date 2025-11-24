# src/agent/knowledge/dataset_loader.py

import os
import pandas as pd
from pathlib import Path
from agent.logging_config import logger
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

DATA_DIR = Path(os.getenv("DATASET_DIR", "datasets"))
SUPPORTED_EXTS = {".csv", ".xlsx", ".xls", ".parquet", ".json"}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def load_dataset(
    filename: str,
    delimiter: str = ",",
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Load CSV, Excel, Parquet, or JSON dataset for PandasAI.
    Automatically detects file type.
    
    Args:
        filename: Name of the dataset file in the datasets folder.
        delimiter: CSV delimiter (default ',').
        encoding: File encoding for CSV (default 'utf-8').

    Returns:
        pd.DataFrame: Loaded DataFrame, empty if error occurs.
    """
    filepath = DATA_DIR / filename

    # Check file existence
    if not filepath.exists():
        logger.error("dataset_missing", filepath=str(filepath))
        return pd.DataFrame()

    ext = filepath.suffix.lower()

    if ext not in SUPPORTED_EXTS:
        logger.error("dataset_unsupported_format", extension=ext)
        return pd.DataFrame()

    try:
        if ext == ".csv":
            df = pd.read_csv(filepath, delimiter=delimiter, encoding=encoding)

        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(filepath)

        elif ext == ".parquet":
            df = pd.read_parquet(filepath)

        elif ext == ".json":
            df = pd.read_json(filepath)

        else:
            logger.error("dataset_unhandled_format", extension=ext)
            return pd.DataFrame()

        logger.info(
            "dataset_loaded",
            filename=filename,
            rows=len(df),
            columns=len(df.columns)
        )
        return df

    except pd.errors.EmptyDataError:
        logger.error("dataset_empty", filename=filename)
        return pd.DataFrame()

    except pd.errors.ParserError as e:
        logger.error("dataset_parse_error", filename=filename, error=str(e))
        return pd.DataFrame()

    except Exception as e:
        logger.exception("dataset_load_error", filename=filename, error=str(e))
        return pd.DataFrame()
