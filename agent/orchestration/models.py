"""Workflow data models, validation errors, and static validation engine."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from agent.core.planner import Plan, PlanStep
from agent.tools.registry import ToolRegistry


class WorkflowValidationError(Exception):
    """Raised when a workflow definition fails structural, dependency, or security validation."""
    pass


class TemplateResolutionError(Exception):
    """Raised when a template expression cannot be safely or validly resolved."""
    pass


@dataclass
class Workflow:
    """Multi-application autonomous workflow specification."""

    id: str
    name: str
    plan: Plan
    description: str = ""
    initial_variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, tool_registry: Optional[ToolRegistry] = None) -> None:
        """Validate workflow structure, dependencies, and template references."""
        WorkflowValidator.validate_workflow(self, tool_registry=tool_registry)


class WorkflowValidator:
    """Static validation engine for cross-application workflows."""

    FORBIDDEN_TEMPLATE_PATTERNS = [
        re.compile(r"__"),
        re.compile(r"\b(eval|exec|import|builtins|globals|locals|system|popen|subprocess|class|mro|bases|subclasses)\b", re.IGNORECASE),
        re.compile(r"[;()]"),
    ]

    TEMPLATE_EXPR_PATTERN = re.compile(r"\{\{([^}]+)\}\}")
    TOKEN_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*$")

    @classmethod
    def validate_workflow(
        cls,
        workflow: Workflow,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        if not workflow.id or not str(workflow.id).strip():
            raise WorkflowValidationError("Workflow must have a non-empty 'id'.")
        if not workflow.name or not str(workflow.name).strip():
            raise WorkflowValidationError("Workflow must have a non-empty 'name'.")

        cls.validate_plan(
            plan=workflow.plan,
            tool_registry=tool_registry,
            initial_variables=workflow.initial_variables,
        )

    @classmethod
    def validate_plan(
        cls,
        plan: Plan,
        tool_registry: Optional[ToolRegistry] = None,
        initial_variables: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not plan or not plan.steps:
            raise WorkflowValidationError("Workflow plan must contain at least one step.")

        step_ids: Set[str] = set()
        step_id_list: List[str] = []

        # 1. Structural inspection & duplicate step IDs
        for idx, step in enumerate(plan.steps):
            sid = step.step_id
            if not sid or not str(sid).strip():
                raise WorkflowValidationError(f"Step at index {idx} is missing a step_id.")
            if sid in step_ids:
                raise WorkflowValidationError(f"Duplicate step_id detected: '{sid}'.")
            step_ids.add(sid)
            step_id_list.append(sid)

            if not step.tool_required or not str(step.tool_required).strip():
                raise WorkflowValidationError(f"Step '{sid}' must specify a required tool.")

            if tool_registry is not None:
                if not tool_registry.has(step.tool_required):
                    raise WorkflowValidationError(
                        f"Step '{sid}' requires tool '{step.tool_required}', which is not registered."
                    )

        # 2. Dependency validation & cycle detection (DAG check)
        graph: Dict[str, List[str]] = {sid: [] for sid in step_ids}
        for step in plan.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    raise WorkflowValidationError(
                        f"Step '{step.step_id}' declares unknown dependency '{dep}'."
                    )
                if dep == step.step_id:
                    raise WorkflowValidationError(
                        f"Step '{step.step_id}' cannot depend on itself."
                    )
                graph[dep].append(step.step_id)

        cls._check_cycles(step_ids, graph)

        # 3. Template reference validation
        declared_vars = set((initial_variables or {}).keys())
        for idx, step in enumerate(plan.steps):
            cls._validate_step_templates(
                step=step,
                available_step_ids=set(step_id_list[:idx]) | set(step.dependencies),
                all_step_ids=step_ids,
                declared_vars=declared_vars,
            )

    @classmethod
    def _check_cycles(cls, step_ids: Set[str], graph: Dict[str, List[str]]) -> None:
        """Verify no cycles exist in step dependency graph using DFS."""
        visited: Dict[str, int] = {sid: 0 for sid in step_ids}

        def dfs(node: str, path: List[str]) -> None:
            visited[node] = 1
            for neighbor in graph.get(node, []):
                if visited[neighbor] == 1:
                    cycle = " -> ".join(path + [neighbor])
                    raise WorkflowValidationError(f"Circular dependency detected in workflow: {cycle}")
                if visited[neighbor] == 0:
                    dfs(neighbor, path + [neighbor])
            visited[node] = 2

        for sid in step_ids:
            if visited[sid] == 0:
                dfs(sid, [sid])

    @classmethod
    def _validate_step_templates(
        cls,
        step: PlanStep,
        available_step_ids: Set[str],
        all_step_ids: Set[str],
        declared_vars: Set[str],
    ) -> None:
        """Scan step arguments for template syntax and injection patterns."""
        def scan_val(val: Any) -> None:
            if isinstance(val, str):
                for match in cls.TEMPLATE_EXPR_PATTERN.finditer(val):
                    expr = match.group(1).strip()
                    for pat in cls.FORBIDDEN_TEMPLATE_PATTERNS:
                        if pat.search(expr):
                            raise WorkflowValidationError(
                                f"Forbidden template expression in step '{step.step_id}': '{expr}'"
                            )

                    if not cls.TOKEN_NAME_PATTERN.match(expr):
                        raise WorkflowValidationError(
                            f"Invalid template token syntax in step '{step.step_id}': '{expr}'"
                        )

                    parts = expr.split(".")
                    root = parts[0]

                    if root in all_step_ids:
                        if root not in available_step_ids:
                            raise WorkflowValidationError(
                                f"Step '{step.step_id}' references output of step '{root}', "
                                f"but '{root}' is not declared as an upstream dependency."
                            )
            elif isinstance(val, dict):
                for k, v in val.items():
                    scan_val(k)
                    scan_val(v)
            elif isinstance(val, list):
                for item in val:
                    scan_val(item)

        scan_val(step.arguments)
