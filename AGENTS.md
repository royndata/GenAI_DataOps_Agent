# Development Rules (AGENTS.md)

This file defines how this project is developed inside Cursor and GitHub.

---

## 1. Coding Rules

- All code lives under `src/agent/`
- Use clear modular boundaries (ingestion, cognition, tools, knowledge, output)
- No monolithic functions > 40 lines
- All external calls must be wrapped (SQL, Slack, PandasAI)
- All subsystems require logging

---

## 2. Git/GitHub Workflow

### Branches:
- **main** → production clean branch (protected)
- **dev** → active development
- **feature/*** branches → every file/feature change

### Workflow:
1. Create feature branch  
2. Make changes  
3. Commit small, atomic commits  
4. Push to feature branch  
5. Create Pull Request → dev  
6. Merge dev → main only after CI passes

---

## 3. Linting & CI Rules

- Ruff + Black for formatting
- Unit tests for each tool subsystem
- GitHub Actions must:
  - Install poetry
  - Run lint
  - Run tests
  - Build Docker image

---

## 4. System Architecture Rules

- Ingestion should never access DB directly  
- Router makes all tool decisions  
- Tools never touch Slack  
- Output module never touches DB  
- Semantic layer is read-only metadata  
- Memory is used only inside cognition

---

## 5. MCP Integration Guidelines

- All external operations must be MCP-tool wrapped:
  - SQL execution  
  - File access  
  - Metadata ops  

- Tools should not call OS/filesystem directly.

---

## 6. Logging Rules

- Ingestion logs queries
- Router logs decisions
- Tools log execution time
- Output logs final response type

---

## 7. Dependency Policy

- Only add packages through `poetry add`
- No local pip installs
- Avoid heavy ML libraries unless required
