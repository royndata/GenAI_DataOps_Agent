# Changelog

All notable changes to the GenAI DataOps Agent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **LLM Reasoning Integration**
  - Added `LLMReasoner` class in `src/agent/cognition/llm_reasoner.py`
  - LLM-based intent interpretation with OpenAI API integration
  - Task decomposition for complex queries
  - Confidence scoring and reasoning explanations
  - Graceful degradation when OpenAI API unavailable

- **Memory System**
  - Added `Memory` class in `src/agent/cognition/memory.py`
  - Short-term conversation history tracking (per user)
  - Long-term context hints storage
  - Conversation retrieval for multi-turn context
  - Automatic cleanup of old conversations

- **Router Enhancements**
  - Integrated LLMReasoner for intelligent intent detection
  - Integrated Memory for conversation context
  - LLM-based routing with fallback to pattern matching
  - Conversation history retrieval before routing
  - Automatic conversation storage after tool execution
  - Settings parameter added to Router initialization

- **Output Guardrails**
  - Added `OutputGuardrails` class in `src/agent/output/output_guardrails.py`
  - Hallucination detection (unrealistic values, inconsistencies)
  - Sensitive data pattern detection (PII, credentials, tokens)
  - SQL injection pattern detection in output
  - Path traversal validation
  - Chart file size validation
  - Message length validation (Slack limits)
  - Data type and range validation
  - Output sanitization before sending to Slack

- **Output Formatter**
  - Added `OutputFormatter` class in `src/agent/output/output_formatter.py`
  - SQL result formatting (tables, row counts, execution time)
  - PandasAI result formatting (analysis summaries, chart indicators)
  - Metric result formatting (formatted values, metadata)
  - Dataset info formatting (schema, row/column counts)
  - Slack message truncation (4000 char limit)
  - Error message formatting
  - Raw data formatting for debugging

- **Input Guardrails Integration**
  - Integrated `InputGuardrails` into `listener.py`
  - Comprehensive validation before routing
  - Unbounded time range detection
  - Unsafe SQL pattern blocking
  - Metric validation using semantic layer
  - Input length and format validation

- **Semantic Layer (Dynamic & Database-Agnostic)**
  - Added `SchemaDiscovery` class in `src/agent/knowledge/schema_discovery.py`
    - Dynamic table discovery via `information_schema`
    - Column metadata discovery
    - Pattern-based table/column matching
    - Schema caching for performance
  - Refactored `semantic_layer.yaml` to use patterns instead of hardcoded names
    - Metric patterns with table/column pattern matching
    - Dataset patterns for PandasAI
    - Routing hints for tool selection
    - Validation rules
  - Added `SemanticLoader` class in `src/agent/knowledge/semantic_loader.py`
    - Maps semantic layer patterns to actual database schema
    - Generates SQL queries from metric definitions
    - Works with any PostgreSQL database dynamically

- **Configuration Management**
  - Migrated from `dataclass` to `pydantic-settings.BaseSettings`
  - Automatic environment variable loading from `.env`
  - Added `SLACK_APP_TOKEN` support
  - Modular database configuration (connection string OR individual fields)
  - Added `.env.example` template with comprehensive documentation

- **Production Enhancements**
  - Enhanced `main.py` with structured startup sequence
  - Component initialization tracking
  - Startup time tracking
  - Graceful shutdown with signal handling
  - Uptime logging
  - Proper exit codes

- **Listener Improvements**
  - Input sanitization (remove bot mentions)
  - Input validation (length, empty checks)
  - Output guardrails integration
  - Output formatter integration
  - Safe chart uploads with error handling
  - Message truncation for Slack limits
  - Structured logging throughout

- **SQL Tool Enhancements**
  - Execution time tracking
  - Query complexity analysis
  - Query length validation
  - Improved error handling
  - Structured result returns

- **PandasAI Tool Enhancements**
  - Thread-safe timeouts using `ThreadPoolExecutor`
  - Explicit chart path management (UUID/timestamp filenames)
  - Token counting with `tiktoken` (optional dependency)
  - Memory monitoring with `psutil` (optional dependency)
  - Enhanced error logging with full tracebacks
  - Custom whitelisted dependencies to reduce logging noise
  - PandasAI v3.x compatibility fixes

### Changed
- **Router Architecture**
  - Router now accepts `settings` parameter for LLMReasoner/Memory initialization
  - Intent detection now uses LLM with pattern matching fallback
  - Added conversation context to routing decisions

- **Listener Architecture**
  - Updated Router initialization to pass `settings` parameter
  - Integrated input guardrails before routing
  - Integrated output guardrails and formatter after routing

- **Configuration**
  - Replaced manual `os.getenv` with `pydantic-settings`
  - Simplified `load_settings()` to return `Settings()` directly

- **Database Layer**
  - `run_query()` now accepts `Union[str, TextClause]` for flexibility
  - SQLTool now passes `TextClause` objects directly to database

### Fixed
- Fixed module import errors (changed `package-mode = false` to `true` in `pyproject.toml`)
- Fixed SQLAlchemy `text()` object handling in `sql_tool.py`
- Fixed PandasAI v3.x import errors (removed deprecated `OpenAI` import)
- Fixed duplicate `handle_reaction` function names in `listener.py`
- Fixed typo in logging filename (`logging_congif.py` → `logging_config.py`)
- Fixed missing `tenacity` dependency
- Fixed `pydantic-settings` import errors

### Security
- Input guardrails prevent unsafe SQL patterns
- Output guardrails prevent sensitive data leaks
- SQL injection prevention in both input and output
- Path traversal prevention
- File size limits for chart uploads
- Message length limits for Slack

## [0.1.0] - 2024-01-XX

### Added
- Initial MVP release
- Core infrastructure (config, database, logging)
- Basic Slack listener
- SQL tool with safety checks
- PandasAI tool integration
- Basic router with pattern matching
- Semantic layer (static YAML)
- Input guardrails
- Main entry point with health checks

### Infrastructure
- Poetry for dependency management
- Structured logging with `structlog`
- Docker support
- Environment variable configuration
- Database connection pooling
- Retry logic for database operations

---

## Notes

- **Version Format**: `[MAJOR.MINOR.PATCH]` following Semantic Versioning
- **Unreleased Section**: Contains all changes since last release
- **Categories**: Added, Changed, Deprecated, Removed, Fixed, Security
- **Date Format**: YYYY-MM-DD

## Future Enhancements (Planned)

- [ ] Human-in-the-loop approval system for high-risk queries
- [ ] MCP (Model Context Protocol) full integration
- [ ] Vector database integration for RAG
- [ ] Advanced LLM reasoning with multi-step workflows
- [ ] Comprehensive test suite
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Performance monitoring and metrics
- [ ] Rate limiting per user/channel
- [ ] Query result caching
- [ ] Multi-database support