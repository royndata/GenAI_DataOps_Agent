# ----------------------------------------------------------
# src/agent/tools/pandasai_tool.py

import os
import time
import uuid
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

from agent.knowledge.dataset_loader import load_dataset
from agent.logging_config import logger
from agent.config import Settings


# Optional deps
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available; token counting disabled")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available; memory monitoring disabled")


# -----------------------------
# PRODUCTION CONSTANTS
# -----------------------------
MAX_DATAFRAME_ROWS = 100000
MAX_EXECUTION_TIME_SECONDS = 60
MAX_CHART_SIZE_MB = 10
MAX_MEMORY_USAGE_MB = 2048
MAX_TOKENS_ESTIMATE = 8000
SUPPORTED_CHART_TYPES = {"bar", "line", "scatter", "histogram", "pie", "box"}

DEFAULT_CHART_DIR = Path("exports/charts")
DEFAULT_CHART_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# ERRORS
# -----------------------------
class PandasAIToolError(Exception):
    pass

class TokenLimitExceededError(PandasAIToolError):
    pass

class MemoryLimitExceededError(PandasAIToolError):
    pass


# -----------------------------
# SLACK-SAFE PARSER
# -----------------------------
class SlackSafeParser:
    """Convert PandasAI output into Slack-safe JSON."""
    @staticmethod
    def parse(result: Any) -> Dict[str, Any]:
        if isinstance(result, pd.DataFrame):
            return {
                "type": "table",
                "columns": list(result.columns),
                "rows": result.to_dict(orient="records"),
                "row_count": len(result),
            }
        if isinstance(result, dict):
            return {"type": "dict", "data": result}
        if isinstance(result, list):
            return {"type": "list", "data": result}
        return {"type": "text", "data": str(result)}


# -----------------------------
# MAIN TOOL
# -----------------------------
class PandasAITool:
    """
    Production-ready PandasAI wrapper with:
    - Slack-safe output
    - Thread-safe timeout
    - Deterministic chart paths
    - Token + memory limits
    - Explicit logging
    """

    def __init__(self, settings: Settings, chart_output_dir: Optional[Union[str, Path]] = None):
        self._validate_api_key(settings.openai_api_key)

        self.llm = OpenAI(
            api_token=settings.openai_api_key,
            temperature=0.1,
            max_tokens=2000,
        )

        self.settings = settings
        self._cache: Dict[str, pd.DataFrame] = {}

        self.chart_output_dir = Path(chart_output_dir) if chart_output_dir else DEFAULT_CHART_DIR
        self.chart_output_dir.mkdir(parents=True, exist_ok=True)

        self.parser = SlackSafeParser()

        self._tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                self._tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except Exception:
                pass

        logger.info("pandasai_tool_initialized", chart_dir=str(self.chart_output_dir))


    # -----------------------------
    # SAFETY HELPERS
    # -----------------------------
    def _validate_api_key(self, api_key: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY missing")

    def _estimate_tokens(self, query: str, df: pd.DataFrame) -> int:
        if self._tokenizer:
            meta = " ".join(df.columns.tolist())
            return len(self._tokenizer.encode(query + meta)) + 500
        return len(query) // 4 + 500

    def _check_token_limit(self, query: str, df: pd.DataFrame):
        if self._estimate_tokens(query, df) > MAX_TOKENS_ESTIMATE:
            raise TokenLimitExceededError("Token estimate exceeded")

    def _check_memory_usage(self) -> float:
        if not PSUTIL_AVAILABLE:
            return 0.0
        mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        if mem > MAX_MEMORY_USAGE_MB:
            raise MemoryLimitExceededError("Memory exceeded")
        return mem

    def _execute_with_timeout(self, func, timeout_seconds, *args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout_seconds)
            except FutureTimeoutError:
                future.cancel()
                raise TimeoutError("LLM execution timeout")


    # -----------------------------
    # DATASET HANDLING
    # -----------------------------
    def _load_or_cache_dataset(self, filename: str) -> pd.DataFrame:
        if filename in self._cache:
            logger.info("pandasai_dataset_cache_hit", filename=filename)
            return self._cache[filename].copy()

        df = load_dataset(filename)
        if df.empty:
            raise ValueError(f"Dataset {filename} empty or missing")

        if len(df) > MAX_DATAFRAME_ROWS:
            df = df.head(MAX_DATAFRAME_ROWS).copy()

        self._cache[filename] = df.copy()
        logger.info("pandasai_dataset_loaded", filename=filename, rows=len(df), columns=len(df.columns))
        return df


    # -----------------------------
    # CHART HELPERS
    # -----------------------------
    def _validate_chart_request(self, chart_type: Optional[str]):
        if chart_type and chart_type.lower() not in SUPPORTED_CHART_TYPES:
            raise ValueError(f"Unsupported chart type {chart_type}")

    def _generate_chart_filename(self, chart_type: str) -> Path:
        ts = int(time.time())
        uid = str(uuid.uuid4())[:8]
        return self.chart_output_dir / f"chart_{ts}_{uid}_{chart_type}.png"


    # -----------------------------
    # MAIN ANALYSIS ENTRYPOINT
    # -----------------------------
    def analyze(
        self,
        query: str,
        dataset_filename: str,
        chart_type: Optional[str] = None,
        save_chart_path: Optional[Union[str, Path]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:

        start = time.time()
        timeout = timeout_seconds or MAX_EXECUTION_TIME_SECONDS

        chart_path = None
        if chart_type:
            chart_path = Path(save_chart_path) if save_chart_path else self._generate_chart_filename(chart_type)

        logger.info("pandasai_analysis_start", query=query[:100], dataset=dataset_filename, chart_type=chart_type)

        try:
            df = self._load_or_cache_dataset(dataset_filename)
            self._validate_chart_request(chart_type)
            self._check_token_limit(query, df)
            self._check_memory_usage()

            smart_df = SmartDataframe(
                df,
                config={
                    "llm": self.llm,
                    "verbose": False,
                    "save_charts": chart_path is not None,
                    "save_charts_path": str(chart_path.parent) if chart_path else None,
                    "custom_whitelisted_dependencies": ["matplotlib", "seaborn"],
                },
            )

            full_query = query if not chart_type else f"{query}. Create a {chart_type} chart and save it to {chart_path.name}"

            result = self._execute_with_timeout(smart_df.chat, timeout, full_query)
            parsed = self.parser.parse(result)

            chart_path_str = None
            if chart_path and chart_path.exists():
                # Validate chart file size
                file_size_mb = chart_path.stat().st_size / (1024 * 1024)
                if file_size_mb > MAX_CHART_SIZE_MB:
                    logger.warning(
                        "pandasai_chart_too_large",
                        path=str(chart_path),
                        size_mb=round(file_size_mb, 2),
                        max_mb=MAX_CHART_SIZE_MB
                    )
                    # Don't include chart_path if it exceeds size limit
                    chart_path_str = None
                else:
                    chart_path_str = str(chart_path.resolve())
                    logger.info(
                        "pandasai_chart_validated",
                        path=str(chart_path),
                        size_mb=round(file_size_mb, 2)
                    )

            execution_time_ms = round((time.time() - start) * 1000, 2)
            logger.info("pandasai_analysis_success", execution_time_ms=execution_time_ms, has_chart=chart_path_str is not None)

            return {
                "success": True,
                "result": parsed,
                "chart_path": chart_path_str,
                "execution_time_ms": execution_time_ms,
                "dataset_rows": len(df),
                "dataset_columns": list(df.columns),
            }

        except (TokenLimitExceededError, MemoryLimitExceededError) as e:
            tb = traceback.format_exc()
            execution_time_ms = round((time.time() - start) * 1000, 2)
            logger.error("pandasai_critical_error", error=str(e), error_type=type(e).__name__, traceback=tb, execution_time_ms=execution_time_ms)
            raise

        except Exception as e:
            tb = traceback.format_exc()
            execution_time_ms = round((time.time() - start) * 1000, 2)
            logger.error("pandasai_analysis_failed", error=str(e), error_type=type(e).__name__, traceback=tb, execution_time_ms=execution_time_ms)
            return {
                "success": False,
                "error": str(e),
                "error_traceback": tb,
                "chart_path": None,
                "result": None,
                "execution_time_ms": execution_time_ms,
            }