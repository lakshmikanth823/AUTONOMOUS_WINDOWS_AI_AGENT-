# Autonomous Windows AI Agent

A production-quality, local-first, modular, provider-independent autonomous AI agent engineered specifically for Windows 10/11.

## Architecture & Principles

* **Local-First Zero-Cost Default**: Leverages local Ollama (`hermes3:8b`) out of the box on `http://127.0.0.1:11434`.
* **Provider-Independent**: Supports Ollama, OpenAI, Anthropic, and Google Gemini via `.env`.
* **Zero Credential Leaks**: API credentials and secrets are masked in CLI representations and never logged.
* **Deterministic Security Tiers**:
  * `SAFE`: Read-only inspections, status probes, non-mutating actions.
  * `LOW_RISK`: Safe local directory file operations and standard development commands.
  * `REQUIRES_APPROVAL`: Process terminations, system modifications, installs, external mutations.
  * `BLOCKED`: Permanently disallowed destructive commands (disk format, drive wiping, recursive system deletion).
* **Deterministic Verification**: Every tool action must verify state changes rather than assuming success.
* **Structured Observability**: Emits dual logs (colored console, rotated human-readable text, and structured JSON logs with `task_id` and `action_id`).

---

## Setup & Installation

### 1. Requirements
* Windows 10 or 11 (64-bit)
* Python 3.11+
* Windows PowerShell 5.1+
* Ollama running locally (optional for local-first execution): `ollama run hermes3:8b`

### 2. Quick Setup

```powershell
# Create dedicated virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install core dependencies
pip install -r requirements.txt

# Copy environment configuration template
copy .env.example .env
```

---

## CLI Usage

Run commands using the project's virtual environment python:

### 1. Interactive Agent Shell
```powershell
.venv\Scripts\python.exe -m agent.main start
```

### 2. Execute a Single Task / Goal
```powershell
.venv\Scripts\python.exe -m agent.main task "Inspect system runtime"
```

### 3. Check System Status & Ollama Availability
```powershell
.venv\Scripts\python.exe -m agent.main status
```

### 4. Inspect Registered Tools & Permissions
```powershell
.venv\Scripts\python.exe -m agent.main tools
```

### 5. Inspect Active Configuration (Secrets Masked)
```powershell
.venv\Scripts\python.exe -m agent.main config
```

### 6. Inspect Memory & Task History
```powershell
.venv\Scripts\python.exe -m agent.main memory
```

---

## Running the Automated Test Suite

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```
All 14 unit tests cover configuration validation, secret masking, permission tiers, command safety classification, task models, tool registration and execution, verification, structured logging, and error handling.
