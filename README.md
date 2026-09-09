# Autonomous Windows AI Agent

A production-quality, local-first, modular, provider-independent autonomous AI agent engineered specifically for Windows 10/11.

## Architecture

```
agent/
├── main.py                     # Entry point & interactive CLI
├── logger.py                   # Structured observable logging (console & rotated JSON)
├── config/
│   ├── settings.py             # Strongly typed Pydantic settings & .env loader
│   └── permissions.py          # Risk tiers, permission scopes, and command safety
├── core/
│   ├── agent.py                # Autonomous agent orchestration loop
│   ├── planner.py              # Goal decomposition & task dependency DAG
│   ├── executor.py             # Tool execution coordinator
│   ├── verifier.py             # Post-action verification assertions
│   ├── recovery.py             # Error taxonomy & self-healing recovery strategies
│   └── state.py                # Agent state machine, task plans, and step history
├── llm/
│   ├── base.py                 # Abstract provider interface & schemas
│   ├── provider.py             # Providers: Ollama (default), OpenAI, Anthropic, Gemini
│   └── prompts.py              # System prompts & few-shot planning instructions
├── tools/
│   ├── registry.py             # Decorator-based tool registry with schema extraction
│   ├── filesystem.py           # Safe read, write, replace, list, find, diff
│   ├── terminal.py             # PowerShell execution with timeout & safety checks
│   ├── browser.py              # Playwright browser automation
│   ├── computer.py             # Windows UI Automation (pywinauto / desktop)
│   └── python_runner.py        # Isolated script execution runner
├── memory/
│   ├── database.py             # SQLite connection & schema management
│   ├── memory_manager.py       # Short-term, long-term, and fact memory coordinator
│   └── schemas.py              # Data models for storage & retrieval
├── security/
│   ├── approval.py             # Human-in-the-loop approval manager
│   ├── sandbox.py              # Path traversal protection & command safety policy
│   └── policy.py               # Risk scoring & authorization rules
├── ui/
│   └── cli.py                  # Rich terminal UI, banners, and interactive prompts
├── tests/                      # Pytest automated test suite
├── data/                       # Local SQLite databases & runtime data
└── logs/                       # Structured JSON and text logs
```

## Key Principles

1. **Python-First & Windows-Native**: Built specifically for Windows 10/11 with PowerShell 5.1/7 compatibility.
2. **Local-First Default**: Works out of the box with locally hosted Ollama models (`hermes3:8b`), requiring zero external API keys.
3. **Provider-Independent**: Easily switch between Ollama, OpenAI, Anthropic Claude, and Google Gemini via `.env`.
4. **Security & Human Governance**: Actions are scored by `RiskLevel` (`READ_ONLY`, `LOW_RISK`, `SENSITIVE`, `DANGEROUS`, `IRREVERSIBLE`). Dangerous and irreversible operations require explicit human confirmation.
5. **Deterministic Verification**: Every tool action has a verification strategy; the agent never assumes success.
6. **Full Observability**: Every execution emits structured logs with `task_id`, `action_id`, timestamps, and detailed payloads.

## Quick Start

### 1. Setup Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```

### 3. Check System Status

```powershell
.venv\Scripts\python.exe -m agent.main info
```

### 4. Run Tests

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```
