"""Unit and integration tests for SQLite persistent memory subsystem."""

from pathlib import Path
import pytest

from agent.config.permissions import PermissionLevel
from agent.core.agent import Agent, TaskState
from agent.core.planner import Plan, Planner, PlanStep
from agent.core.state import StepResult, TaskStatus
from agent.llm.provider import MockLLMProvider
from agent.memory.manager import MemoryManager
from agent.memory.schemas import (
    MemoryCategory,
    MemoryRecord,
    sanitize_content,
)
from agent.memory.store import MemoryStore
from agent.tools.base import Tool, ToolResult
from agent.tools.registry import ToolRegistry


# ==============================================================================
# 1. Memory CRUD Operations Tests
# ==============================================================================

def test_memory_crud_operations(tmp_path: Path):
    """Verify complete CRUD lifecycle of memory records in SQLite."""
    db_file = tmp_path / "test_memory.db"
    store = MemoryStore(db_path=db_file)

    # 1. Create / Save
    rec = MemoryRecord(
        category=MemoryCategory.FACT,
        content="Host runs Windows 11 with PowerShell 5.1",
        importance=0.8,
        metadata={"os": "win11"},
    )
    saved_id = store.save(rec)
    assert saved_id == rec.id

    # 2. Read / Retrieve
    fetched = store.get(saved_id)
    assert fetched is not None
    assert fetched.content == "Host runs Windows 11 with PowerShell 5.1"
    assert fetched.category == MemoryCategory.FACT
    assert fetched.importance == 0.8
    assert fetched.metadata["os"] == "win11"

    # 3. Update
    updated = store.update(saved_id, content="Host runs Windows 11 64-bit", importance=0.9)
    assert updated is True
    refetched = store.get(saved_id)
    assert refetched.content == "Host runs Windows 11 64-bit"
    assert refetched.importance == 0.9

    # 4. List by category
    facts = store.list_by_category(MemoryCategory.FACT)
    assert len(facts) == 1

    # 5. Delete
    deleted = store.delete(saved_id)
    assert deleted is True
    assert store.get(saved_id) is None


# ==============================================================================
# 2. Secret Sanitization Tests
# ==============================================================================

def test_secret_sanitization(tmp_path: Path):
    """Enforce: NEVER store API keys, tokens, or passwords in plain text in memory."""
    db_file = tmp_path / "test_secrets.db"
    store = MemoryStore(db_path=db_file)

    leaked_content = (
        "Configured OpenAI with sk-abcdef123456789012345678 and "
        "password='super_secret_pass_999' and bearer my_auth_token_123456789"
    )

    # Sanitize check
    clean = sanitize_content(leaked_content)
    assert "sk-abcdef123456789012345678" not in clean
    assert "super_secret_pass_999" not in clean
    assert "[REDACTED_CREDENTIAL]" in clean

    # Storage check: saving via store must automatically sanitize
    rec = MemoryRecord(
        category=MemoryCategory.CONVERSATION,
        content=leaked_content,
    )
    saved_id = store.save(rec)
    stored_rec = store.get(saved_id)

    assert "sk-abcdef123456789012345678" not in stored_rec.content
    assert "super_secret_pass_999" not in stored_rec.content
    assert "[REDACTED_CREDENTIAL]" in stored_rec.content


# ==============================================================================
# 3. Relevance Search Tests
# ==============================================================================

def test_relevance_search(tmp_path: Path):
    """Verify relevance-based retrieval using FTS5 search."""
    db_file = tmp_path / "test_search.db"
    store = MemoryStore(db_path=db_file)

    # Insert diverse records
    store.save(MemoryRecord(
        category=MemoryCategory.USER_PREFERENCE,
        content="Always use Edge browser for automation tasks",
        importance=0.9,
    ))
    store.save(MemoryRecord(
        category=MemoryCategory.FACT,
        content="Python 3.11 virtual environment is located in .venv",
        importance=0.8,
    ))
    store.save(MemoryRecord(
        category=MemoryCategory.TASK,
        content="Created database schema for user analytics",
        importance=0.5,
    ))

    # Search for browser preference
    browser_hits = store.search("Edge browser")
    assert len(browser_hits) >= 1
    assert "Edge browser" in browser_hits[0].record.content

    # Search for python
    python_hits = store.search("Python virtual environment")
    assert len(python_hits) >= 1
    assert ".venv" in python_hits[0].record.content


# ==============================================================================
# 4. High-Level MemoryManager Tests
# ==============================================================================

def test_memory_manager_workflows(tmp_path: Path):
    """Verify user preferences, facts, and task completion recording."""
    db_file = tmp_path / "test_manager.db"
    store = MemoryStore(db_path=db_file)
    mgr = MemoryManager(store=store)

    # Record User Preference
    pref_id = mgr.record_user_preference("color_scheme", "dark_mode")
    assert pref_id.startswith("mem_")

    # Record Learned Fact
    fact_id = mgr.record_learned_fact("PowerShell 5.1 requires Bypass execution policy")
    assert fact_id.startswith("mem_")

    # Record Task Completion
    task_state = TaskState(
        user_goal="Setup workspace",
        status=TaskStatus.COMPLETED,
        observations=[
            StepResult(
                tool_name="filesystem",
                arguments={"action": "create_directory", "path": "test"},
                success=True,
                output="Created",
                verification_passed=True,
            )
        ],
    )
    task_mem_id = mgr.record_task_completion(task_state)
    assert task_mem_id.startswith("mem_")

    # Verify task summary
    summary = mgr.store.summarize_tasks()
    assert "Setup workspace" in summary
    assert "COMPLETED" in summary

    # Verify relevant context extraction
    context = mgr.get_relevant_context("PowerShell")
    assert "PowerShell 5.1" in context


# ==============================================================================
# 5. Agent Loop Memory Context Integration Tests
# ==============================================================================

def test_agent_memory_injection_in_planning(tmp_path: Path):
    """Verify that Agent retrieves relevant memories and injects them into Planner."""
    db_file = tmp_path / "test_agent_mem.db"
    store = MemoryStore(db_path=db_file)
    mgr = MemoryManager(store=store)

    # Pre-seed memory with an important fact
    mgr.record_learned_fact("Always verify file size before reporting completion.")

    plan_json = '{"goal": "Create file", "steps": [{"step_id": "step_1", "objective": "Create file", "tool_required": "echo", "arguments": {"message": "done"}}]}'
    mock_provider = MockLLMProvider(responses=[plan_json])
    planner = Planner(provider=mock_provider)

    class EchoTool(Tool):
        name = "echo"
        description = "Echo"
        permission_level = PermissionLevel.SAFE
        input_schema = {"type": "object", "properties": {}}
        def execute(self, args):
            return ToolResult(success=True, output="done")

    reg = ToolRegistry()
    reg.register(EchoTool())

    agent = Agent(planner=planner, tool_registry=reg, memory_manager=mgr)
    state = agent.run("Create file")

    assert state.status == TaskStatus.COMPLETED

    # Verify that the LLM call history received the memory context
    call_prompt = mock_provider.call_history[0]
    assert "Relevant Memories from Previous Tasks" in call_prompt
    assert "Always verify file size" in call_prompt

    # Verify that the completed task was stored in memory
    task_memories = store.list_by_category(MemoryCategory.TASK)
    assert len(task_memories) == 1
    assert "Create file" in task_memories[0].content
