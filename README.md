# GenAI DataOps Agent

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/PandasAI-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-LLM-green)
![Slack](https://img.shields.io/badge/Slack-Bot-purple)

A **production-grade, modular GenAI analytics agent** that processes natural-language analytical questions, routes them intelligently through SQL/PandasAI/LLM tools, applies guardrails, and returns formatted insights (tables, metrics, charts).  

Designed to work with **Slack (default)** but fully interface-agnostic — can be swapped for **API, Teams, Email, CLI, nodeJS.**.

This project is built using DataOps/GenAI principles, designed for **real-world enterprise deployment** with Docker, GitHub Actions, MCP integration, and extensible data toolchains.

---

## 1. Features

### 🔹 Universal Analytics Engine  
- Query SQL databases  
- Run PandasAI dataframe analysis + charts  
- Use semantic layer for accurate metric interpretation  
- LLM reasoning with memory + guardrails  
- Subsystem routing for intelligent tool dispatching  

### 🔹 Interface-Agnostic 
Designed so the ingestion/output interface can be switched with:
- Slack  
- Email / Gmail  
- API Gateway  
- Teams  
- Command Line  
- React / Next.js frontend 

### 🔹 Production-Ready Components  
- Fully modular code structure (src/agent/…)  
- Clean subsystem boundaries  
- Cognitive loop + router  
- Input and output guardrails  
- Tools engine + knowledge store separation  

### 🔹 DevOps-Ready  
- Dockerfile included  
- GitHub Actions-compatible structure  
- Poetry environment  
- Local + Cloud version supported  
- MCP client ready for SQL, metadata, and file operations  

---
## 2. Core Capabilities

This agent supports:

### 🔹 Flexible Input Interfaces
Works with Slack today, but easily replaceable with:
- REST API
- Teams
- Email
- CLI
- React / Next.js UI

### 🔹 Input Guardrails
Prevents dangerous, impossible, or system-breaking queries:
- Rejects unbounded time ranges (“all logs for 3 years”)  
- Detects unsafe SQL patterns  
- Ensures questions match available metrics  
- Normalizes text before routing  

### 🔹 Cognition Engine
A modular “thinking loop”:
- **Router** — decides which tool to use
- **LLM Reasoner** — interprets intent, decomposes complex tasks
- **Memory** — short-term (conversation) + long-term (context hints)

### 🔹 Tools Engine
Executes actions:
- **SQL Tool** — fast structured queries
- **PandasAI Tool** — charts, visual analytics, transformations
- **Retriever Tool (stub)** — for future RAG + vector DB
- **MCP Client** — bridges external capabilities

### 🔹 Knowledge Store
The system’s “truth”:
- **Semantic Layer** → metric definitions  
- **Datasets** → CSV/Parquet for PandasAI  
- **Database** → Postgres (Render / local)

### 🔹 Output Layer
- Output formatter (tables, summaries, charts)
- Output guardrails (safety + correctness)
- Slack message response  
---

Architecture diagram:

```markdown
═══════════════════════════════════════════════════════════════════════════════════════
                    GENAI DATAOPS AGENT - SYSTEM ARCHITECTURE DIAGRAM
═══════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL INTERFACES                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ User Queries
                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  [1] INGESTION SUBSYSTEM                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │  SlackListener                                                               │   │
│  │  • Receives @mentions, DMs, reactions                                        │   │
│  │  • Input sanitization                                                        │   │
│  │  • Sends formatted responses                                                 │   │
│  │  • Chart uploads                                                             │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ (one-way: input)
                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  [7] INPUT GUARDRAILS                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │  InputGuardrails                                                             │   │
│  │  • Unbounded time range validation                                           │   │
│  │  • Unsafe SQL pattern detection                                              │   │
│  │  • Metric validation                                                         │   │
│  │  • Input length/format checks                                                │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ (one-way: validated input)
                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  [2] COGNITION SUBSYSTEM                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │  Router                                                                      │   │
│  │  • Rate limiting (per user)                                                  │   │
│  │  • Intent detection                                                          │   │
│  │  • Tool selection                                                            │   │
│  │  • NL → SQL conversion                                                       │   │
│  │  • Query complexity analysis                                                 │   │
│  │  • Date filter extraction                                                    │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                      │
│         ┌────────────────────┼────────────────────┐                                 │
│         │                    │                    │                                 │
│         │ (↔ bidirectional)  │ (↔ bidirectional) │                                  │
│         │                    │                    │                                 │
│         ▼                    ▼                    ▼                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                           │
│  │ LLMReasoner  │    │   Memory     │    │SemanticLoader│                           │
│  │              │    │              │    │              │                           │
│  │ • Intent     │◄───│ • Conversation│    │ • Metric     │                          │
│  │   interpret  │    │   history    │    │   mapping    │                           │
│  │ • Task       │    │ • Context    │    │ • SQL        │                           │
│  │   decompose  │    │   hints      │    │   generation │                           │
│  │ • NL→SQL     │    │ • Per-user   │    │ • Schema     │                           │
│  │   conversion │    │   storage    │    │   discovery  │                           │
│  └──────────────┘    └──────────────┘    └──────────────┘                           │
│         │                    │                    │                                 │
│         │                    │                    │                                 │
│         └────────────────────┴────────────────────┘                                 │
│                              │                                                      │
│                              │ (one-way: router → semantic)                         │
│                              ▼                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ (one-way: router → tools)
                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  [4] KNOWLEDGE SUBSYSTEM                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │  SemanticLoader                                                              │   │
│  │  • Loads semantic_layer.yaml                                                 │   │
│  │  • Maps metrics to schema                                                    │   │
│  │  • Generates complex SQL                                                     │   │
│  │                                                                              │   │
│  │  ┌──────────────────────────────────────────────────────────────────────┐    │   │
│  │  │  SchemaDiscovery                                                     │    │   │
│  │  │  • Dynamic table/column discovery                                    │    │   │
│  │  │  • Schema caching                                                    │    │   │
│  │  │  • Pattern matching                                                  │    │   │
│  │  └──────────────────────────────────────────────────────────────────────┘    │   │
│  │         │                                                                    │   │
│  │         │ (↔ bidirectional: semantic ↔ schema)                               │   │
│  │         │                                                                    │   │
│  │         ▼                                                                    │   │
│  │  ┌──────────────────────────────────────────────────────────────────────┐    │   │
│  │  │  Database                                                            │    │   │
│  │  │  • SQLAlchemy connection pooling                                     │    │   │
│  │  │  • Query execution                                                   │    │   │
│  │  │  • Transaction safety                                                │    │   │
│  │  │  • Retry logic                                                       │    │   │
│  │  └──────────────────────────────────────────────────────────────────────┘    │   │
│  │         │                                                                    │   │
│  │         │ (↔ bidirectional: schema ↔ database)                               │   │
│  │         │                                                                    │   │
│  │  DatasetLoader                                                               │   │
│  │  • CSV/Parquet file loading                                                  │   │
│  │  • Caching                                                                   │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ (one-way: knowledge → tools)
                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  [3] TOOLS SUBSYSTEM                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │  SQLTool                                                                     │   │
│  │  • Safe SELECT-only queries                                                  │   │
│  │  • Query validation & normalization                                          │   │
│  │  • Row limits (5000 max)                                                     │   │
│  │  • Execution time tracking                                                   │   │
│  │                                                                              │   │
│  │  PandasAITool                                                                │   │
│  │  • Data analysis & visualization                                             │   │
│  │  • Chart generation                                                          │   │
│  │  • Token/memory monitoring                                                   │   │
│  │                                                                              │   │
│  │  RetrieverTool (STUB)                                                        │   │
│  │  MCPClient (STUB)                                                            │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ (one-way: tools → output guardrails)
                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  [7] OUTPUT GUARDRAILS                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │  OutputGuardrails                                                            │   │
│  │  • Hallucination detection                                                   │   │
│  │  • Sensitive data filtering                                                  │   │
│  │  • Format validation                                                         │   │
│  │  • Chart path validation                                                     │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ (one-way: validated output)
                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  [6] OUTPUT FORMATTER                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │  OutputFormatter                                                             │   │
│  │  • Formats SQL results (markdown tables)                                     │   │
│  │  • Formats PandasAI results                                                  │   │
│  │  • Formats metric results                                                    │   │
│  │  • Slack message truncation (4000 char limit)                                │   │
│  │  • Decimal/currency formatting                                               │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ (one-way: formatted output)
                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  [1] INGESTION SUBSYSTEM (OUTPUT)                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │  SlackListener                                                               │   │
│  │  • Sends formatted message to Slack                                          │   │
│  │  • Uploads charts (if generated)                                             │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ Response
                              ▼
                    ┌─────────────────────┐
                    │   Slack User        │
                    └─────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════
                              DATA FLOW SUMMARY
═══════════════════════════════════════════════════════════════════════════════════════

ONE-WAY FLOWS (→):
─────────────────
1. Ingestion → InputGuardrails
2. InputGuardrails → Router
3. Router → SemanticLoader (read metadata)
4. Router → SQLTool / PandasAITool
5. SQLTool → Database (queries)
6. Tools → OutputGuardrails
7. OutputGuardrails → OutputFormatter
8. OutputFormatter → Ingestion (output)
9. Memory → LLMReasoner (context only)

BIDIRECTIONAL FLOWS (↔):
────────────────────────
1. Router ↔ LLMReasoner
   • Router requests intent interpretation
   • LLMReasoner returns intent/confidence
   • Router requests NL→SQL conversion
   • LLMReasoner returns SQL query

2. Router ↔ Memory
   • Router stores conversations (write)
   • Router retrieves conversation history (read)

3. SemanticLoader ↔ SchemaDiscovery
   • SemanticLoader queries SchemaDiscovery for schema mapping
   • SchemaDiscovery provides table/column information

4. SchemaDiscovery ↔ Database
   • SchemaDiscovery queries Database for schema info
   • Database returns schema metadata

═══════════════════════════════════════════════════════════════════════════════════════
                              EXTERNAL DEPENDENCIES
═══════════════════════════════════════════════════════════════════════════════════════

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Slack API   │    │  OpenAI API  │    │  PostgreSQL  │    │  PandasAI    │
│  (Socket     │    │  (LLM        │    │  Database    │    │  (Analytics) │
│   Mode)      │    │   calls)     │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
      │                   │                   │                   │
      │                   │                   │                   │
      └───────────────────┴───────────────────┴───────────────────┘
                          │
                          │
                  ┌─────────────────┐
                  │  GenAI Agent    │
                  │  (This System)  │
                  └─────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════
                              SUBSYSTEM SUMMARY
═══════════════════════════════════════════════════════════════════════════════════════

[1] INGESTION:        SlackListener
[2] COGNITION:        Router, LLMReasoner, Memory
[3] TOOLS:            SQLTool, PandasAITool, RetrieverTool, MCPClient
[4] KNOWLEDGE:        SemanticLoader, SchemaDiscovery, Database, DatasetLoader
[5] MEMORY:           Memory (separate subsystem, used by Cognition)
[6] OUTPUT FORMATTER: OutputFormatter
[7] GUARDRAILS:       InputGuardrails, OutputGuardrails

═══════════════════════════════════════════════════════════════════════════════════════
                              MVP STATUS: ✅ YES
═══════════════════════════════════════════════════════════════════════════════════════

Core Features Implemented:
✓ Natural language query processing
✓ Intelligent routing (SQL/PandasAI/LLM)
✓ Semantic layer integration
✓ Complex SQL generation (JOINs, GROUP BY, HAVING, subqueries)
✓ Query complexity analysis
✓ Date filter building
✓ Input/output guardrails
✓ Memory/conversation context
✓ Human-in-the-loop confirmations
✓ Output formatting (tables, charts)
✓ Production-ready structure (Docker, modular, testable)
✓ Error handling & logging
✓ Rate limiting
✓ SQL normalization & validation

Ready for Production: ✅ YES
```

---
## 4. 📂 Project Structure

```
project-root/
├── README.md
├── AGENTS.md
├── PROJECT_CONTEXT.md
├── CHANGELOG.md
├── .env
├── pyproject.toml
├── poetry.lock
├── Dockerfile
├── .dockerignore
├── .gitignore
│
├── docs/
│   ├── Data_Flow.png
│   └── slack_agent.png
│
├── exports/
│   └── charts/
│
├── test/
│   ├── test_e2e.py
│   ├── test.md
│   └── verify_infra.py
│
└── src/
    └── agent/
        ├── main.py
        ├── config.py
        ├── health.py
        ├── logging_config.py
        │
        ├── ingestion/
        │   ├── __init__.py
        │   ├── listener.py
        │   └── input_guardrails.py
        │
        ├── cognition/
        │   ├── __init__.py
        │   ├── router.py
        │   ├── llm_reasoner.py
        │   └── memory.py
        │
        ├── tools/
        │   ├── __init__.py
        │   ├── sql_tool.py
        │   ├── pandasai_tool.py
        │   ├── retriever_tool.py
        │   └── mcp_client.py
        │
        ├── knowledge/
        │   ├── __init__.py
        │   ├── semantic_layer.yaml
        │   ├── semantic_loader.py
        │   ├── schema_discovery.py
        │   ├── database.py
        │   ├── dataset_loader.py
        │   ├── complex_sql_generator.py
        │   ├── query_analyzer.py
        │   ├── date_filter_builder.py
        │   ├── prompt_manager.py
        │   └── sql_validator.py
        │
        └── output/
            ├── __init__.py
            ├── output_formatter.py
            └── output_guardrails.py
```

---

## 5. 🛠️ Technologies Used

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 |
| Package Manager | Poetry |
| AI/LLM | OpenAI / LiteLLM |
| Analytics | PandasAI |
| Interface | Slack Bolt SDK |
| Database | Postgres (Render / local) |
| Packaging | Docker |
| Dev Safety | Ruff + Black |
| Future Expansion | MCP, RAG, S3, Vector DB |

## 6.🔒 Guardrails (Safety Layer)

The agent includes:

Input Guardrails
Prevent long-running queries, unsafe requests, unbounded date ranges, invalid metrics.

Output Guardrails
Prevent hallucinations, unsafe content, and formatting issues.

---

## 7. Quick Start

- Install dependencies
- Activate environment
- Add environment vars:
  - SLACK_BOT_TOKEN=
  - SLACK_SIGNING_SECRET=
  - OPENAI_API_KEY=
  - DB_CONNECTION_STRING=
- Run the agent
  - python src/agent/main.py

---

## 8. Testing

Tests will live under `/test` and cover:

- Router logic
- Guardrails
- SQL and PandasAI tool dispatch
- End-to-end Slack message simulation

---

## 9. Roadmap

### Production Essentials
- [ ] **CI/CD Pipeline** - GitHub Actions for automated testing & deployment
- [ ] **Testing Suite** - Unit, integration, and E2E tests
- [ ] **Monitoring & Observability** - Metrics, logging, alerting
- [ ] **Query Caching** - Redis-based result caching for performance

### High-Value Features
- [ ] **RAG + VectorDB** - Retriever Tool with context-aware responses
- [ ] **Multi-Database Support** - MySQL, BigQuery, Snowflake beyond PostgreSQL
- [ ] **S3/Athena Integration** - Cloud data source support
- [ ] **Multi-Interface Support** - Teams, Web API, Email beyond Slack

### Enterprise Readiness
- [ ] **Authentication & Authorization** - RBAC, API keys, user management
- [ ] **Security Hardening** - PII detection, encryption, audit logging
- [ ] **UI Dashboard** - Query management & analytics interface
- [ ] **Benchmarking Suite** - Performance evaluation & optimization
