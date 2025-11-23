# src/agent/knowledge/dataset_loader.py

import yaml
import pandas as pd
from pathlib import Path
from agent.logging_config import logger


class DatasetLoader:
    """
    Handles:
    - semantic layer loading (YAML)
    - local CSV/Parquet dataset loading
    """

    def __init__(self, semantic_layer_path: str = None):
        self.semantic_layer_path = semantic_layer_path or str(
            Path(__file__).parent / "semantic_layer.yaml"
        )
        self.semantic_layer = self._load_semantic_layer()

    def _load_semantic_layer(self) -> dict:
        """
        Loads semantic metadata for metrics mapping.
        """
        try:
            with open(self.semantic_layer_path, "r") as f:
                data = yaml.safe_load(f)
            logger.info("semantic_layer_loaded")
            return data
        except Exception as e:
            logger.error("semantic_layer_failed", error=str(e))
            return {}

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Loads local CSV/Parquet files used by PandasAI.
        """
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.endswith(".parquet"):
                df = pd.read_parquet(file_path)
            else:
                raise ValueError("Unsupported file format")

            logger.info("dataset_loaded", file=file_path)
            return df

        except Exception as e:
            logger.error("dataset_loading_failed", file=file_path, error=str(e))
            raise
