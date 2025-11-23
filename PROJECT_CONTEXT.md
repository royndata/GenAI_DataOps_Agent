# Project Context

This document captures **the reasoning, decisions, and constraints** behind the GenAI DataOps Agent.

---

## 1. Project Purpose

Build a **universal analytics agent** that:
- Understands business questions
- Routes intelligently between SQL, PandasAI, and LLM reasoning
- Is modular, production-capable, and easily extendable
- Can integrate future cloud services (S3, Athena, VectorDB, MCP tools)

---

## 2. Versioning Strategy

We build in two phases:

### **Version A — Local / SQL** (current)
- Local Postgres
- PandasAI for charts & EDA
- Slack as first interface (but interchangeable)
- MCP used for SQL + file tools
- Full modular subsystem architecture

### **Version B — Cloud / S3** (later)
- S3 datasets
- Athena SQL
- VectorDB retriever
- Full RAG capabilities
- Cloud-based MCP tools

---

## 3. System Architecture (High-Level)

### Subsystems:
1. **Ingestion**  
   Converts interface → clean text  
   Guardrails prevent invalid queries.

2. **Cognition Engine**  
   Router, Reasoner, Memory.

3. **Tools Engine**  
   SQL Tool  
   PandasAI Tool  
   Retriever Tool (stub)  
   MCP Client

4. **Knowledge Store**  
   Semantic layer  
   Local datasets  
   PostgreSQL

5. **Output Module**  
   Formatter + Output guardrails.

---

## 4. Routing Logic

The router selects tools based on:
- Semantic layer (metric definitions)
- Query type (metric vs analytic vs free text)
- LLM intent classification

Examples:
- “Show revenue last 30 days” → SQL
- “Plot DAU vs churn trend” → PandasAI
- “Summarize user behavior patterns” → LLM reasoning

---

## 5. Slack Independence

Slack is *just* an ingestion/output interface.  
The bot can be switched to:
- FastAPI
- Microsoft Teams
- Discord
- Email
- CLI  
with zero architecture change.

---

## 6. Constraints

- Must run in Python 3.11
- Must use Poetry
- Must be fully modular
- Must support Docker & CI/CD
- Must support chart/image output

---

## 7. Future Extensions

- S3 ingestion
- Athena SQL
- VectorDB retriever
- LLM memory store
- Multi-agent orchestration
