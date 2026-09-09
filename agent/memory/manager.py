"""MemoryManager coordinating high-level memory operations, relevance retrieval, and task history."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.core.agent import TaskState
from agent.memory.schemas import MemoryCategory, MemoryRecord
from agent.memory.store import MemoryStore


class MemoryManager:
    """High-level coordinator for memory persistence, retrieval, and contextual relevance."""

    def __init__(self, store: Optional[MemoryStore] = None) -> None:
        self.store = store or MemoryStore()

    def add_memory(
        self,
        category: MemoryCategory,
        content: str,
        source: str = "agent",
        importance: float = 0.5,
        task_id: Optional[str] = None,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create and save a new memory record."""
        record = MemoryRecord(
            category=category,
            content=content,
            source=source,
            importance=importance,
            task_id=task_id,
            project_id=project_id,
            metadata=metadata or {},
        )
        return self.store.save(record)

    def record_user_preference(self, key: str, value: str) -> str:
        """Store or update persistent user preference."""
        content = f"User preference: '{key}' = '{value}'"
        return self.add_memory(
            category=MemoryCategory.USER_PREFERENCE,
            content=content,
            source="user",
            importance=0.9,
            metadata={"preference_key": key, "preference_value": value},
        )

    def record_learned_fact(self, fact: str, source: str = "agent") -> str:
        """Record an important learned fact discovered during execution."""
        return self.add_memory(
            category=MemoryCategory.FACT,
            content=fact,
            source=source,
            importance=0.8,
        )

    def record_project_context(self, project_id: str, context: str) -> str:
        """Record project-specific knowledge or configuration."""
        return self.add_memory(
            category=MemoryCategory.PROJECT,
            content=context,
            source="project_inspector",
            importance=0.8,
            project_id=project_id,
        )

    def record_task_completion(self, state: TaskState) -> str:
        """Record a completed or failed task run and significant observations."""
        status_str = getattr(state.status, "value", str(state.status))
        obs_count = len(state.observations)
        steps_count = len(state.plan.steps) if state.plan else 0

        summary = (
            f"Task '{state.user_goal}' concluded with status {status_str}. "
            f"Executed {obs_count} actions across {steps_count} plan steps."
        )
        if state.errors:
            summary += f" Errors: {'; '.join(state.errors[:2])}"

        task_mem_id = self.add_memory(
            category=MemoryCategory.TASK,
            content=summary,
            source="agent_orchestrator",
            importance=0.7,
            task_id=state.task_id,
            metadata={
                "status": status_str,
                "goal": state.user_goal,
                "errors": state.errors,
                "steps_completed": obs_count,
            },
        )

        # Store critical tool actions
        for obs in state.observations:
            if obs.success:
                self.add_memory(
                    category=MemoryCategory.TOOL_ACTION,
                    content=f"Tool '{obs.tool_name}' executed: {obs.arguments}. Result: {str(obs.output)[:150]}",
                    source="tool_executor",
                    importance=0.4,
                    task_id=state.task_id,
                    metadata={"tool": obs.tool_name, "verified": obs.verification_passed},
                )

        return task_mem_id

    def get_relevant_context(self, query: str, limit: int = 5) -> str:
        """Retrieve relevant memories matching query without dumping the full database."""
        # Search across facts, preferences, project context, and past tasks
        results = self.store.search(query=query, limit=limit)
        if not results:
            # Fallback: check user preferences if any exist
            prefs = self.store.list_by_category(MemoryCategory.USER_PREFERENCE, limit=3)
            if prefs:
                lines = ["Relevant Preferences:"]
                for p in prefs:
                    lines.append(f"- {p.content}")
                return "\n".join(lines)
            return ""

        lines = ["Relevant Memories from Previous Tasks:"]
        for res in results:
            cat_label = res.record.category.value.replace("_", " ").title()
            lines.append(f"- [{cat_label}] {res.record.content}")
        return "\n".join(lines)
