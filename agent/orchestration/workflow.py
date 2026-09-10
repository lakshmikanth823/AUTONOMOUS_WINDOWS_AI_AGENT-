"""Cross-application workflow orchestrator and context pipeline."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from agent.config.settings import Settings, get_settings
from agent.core.agent import Agent
from agent.core.planner import Plan, PlanStep, Planner
from agent.core.state import TaskState, TaskStateEnum
from agent.core.verifier import Verifier, default_verifier
from agent.memory import default_memory_manager
from agent.memory.manager import MemoryManager
from agent.orchestration.models import (
    TemplateResolutionError,
    Workflow,
    WorkflowValidationError,
    WorkflowValidator,
)
from agent.tools.base import ToolResult
from agent.tools.registry import ToolRegistry, registry

logger = logging.getLogger(__name__)


class StrictTemplateResolver:
    """Secure, sandboxed template resolver for cross-step data piping.

    Invariants:
    - Absolutely NO eval(), exec(), or dynamic code evaluation.
    - Untrusted data boundary: all resolved values are treated purely as data.
    - Type preservation: exact expression {{...}} preserves raw types (dict, int, bool, list).
    - Blocks dunder attributes (__class__, __mro__, etc.) and forbidden builtins.
    """

    EXACT_TEMPLATE_REGEX = re.compile(r"^\{\{([^}]+)\}\}$")
    SUB_TEMPLATE_REGEX = re.compile(r"\{\{([^}]+)\}\}")
    TOKEN_REGEX = re.compile(r"^[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*$")
    FORBIDDEN_IDENTIFIERS = {
        "eval", "exec", "import", "globals", "locals", "builtins",
        "system", "popen", "subprocess", "class", "mro", "bases", "subclasses"
    }

    @classmethod
    def validate_expression(cls, expr: str) -> None:
        expr = expr.strip()
        if not cls.TOKEN_REGEX.match(expr):
            raise TemplateResolutionError(f"Malformed template expression syntax: '{expr}'")
        parts = expr.split(".")
        for part in parts:
            if "__" in part or part.lower() in cls.FORBIDDEN_IDENTIFIERS or part.startswith("_"):
                raise TemplateResolutionError(f"Security violation: access to forbidden token '{part}' in '{expr}'")

    @classmethod
    def resolve_token(
        cls,
        expr: str,
        step_outputs: Dict[str, Any],
        variables: Dict[str, Any],
        allow_unresolved: bool = False,
    ) -> Any:
        cls.validate_expression(expr)
        parts = expr.strip().split(".")
        root = parts[0]

        if root in step_outputs:
            val = step_outputs[root]
            remaining = parts[1:]
            # If remaining starts with 'output', skip it if val doesn't have an explicit 'output' key
            if remaining and remaining[0] == "output":
                if isinstance(val, dict) and "output" in val:
                    val = val["output"]
                remaining = remaining[1:]

            for p in remaining:
                if val is None:
                    return None
                if isinstance(val, dict):
                    if p in val:
                        val = val[p]
                    else:
                        raise TemplateResolutionError(f"Key '{p}' not found in step output for '{root}'")
                elif isinstance(val, (list, tuple)):
                    try:
                        idx = int(p)
                        val = val[idx]
                    except (ValueError, IndexError):
                        raise TemplateResolutionError(f"Invalid index '{p}' in list for '{root}'")
                elif hasattr(val, p):
                    if p.startswith("_"):
                        raise TemplateResolutionError(f"Access to private attribute '{p}' is forbidden")
                    val = getattr(val, p)
                else:
                    raise TemplateResolutionError(f"Attribute or key '{p}' not found on '{root}'")
            return val

        if root in variables:
            val = variables[root]
            for p in parts[1:]:
                if val is None:
                    return None
                if isinstance(val, dict):
                    if p in val:
                        val = val[p]
                    else:
                        raise TemplateResolutionError(f"Key '{p}' not found in variable '{root}'")
                elif isinstance(val, (list, tuple)):
                    try:
                        idx = int(p)
                        val = val[idx]
                    except (ValueError, IndexError):
                        raise TemplateResolutionError(f"Invalid index '{p}' in list for '{root}'")
                elif hasattr(val, p):
                    if p.startswith("_"):
                        raise TemplateResolutionError(f"Access to private attribute '{p}' is forbidden")
                    val = getattr(val, p)
                else:
                    raise TemplateResolutionError(f"Attribute or key '{p}' not found on variable '{root}'")
            return val

        if allow_unresolved:
            return f"{{{{{expr}}}}}"

        raise TemplateResolutionError(f"Reference '{root}' not found in step outputs or variables.")

    @classmethod
    def resolve_value(
        cls,
        data: Any,
        step_outputs: Dict[str, Any],
        variables: Dict[str, Any],
        allow_unresolved: bool = False,
    ) -> Any:
        if isinstance(data, str):
            # Check for exact scalar match first for type preservation
            exact_m = cls.EXACT_TEMPLATE_REGEX.match(data.strip())
            if exact_m:
                return cls.resolve_token(
                    exact_m.group(1).strip(),
                    step_outputs,
                    variables,
                    allow_unresolved=allow_unresolved,
                )

            # Embedded replacement within string
            def replace_fn(match: re.Match) -> str:
                resolved = cls.resolve_token(
                    match.group(1).strip(),
                    step_outputs,
                    variables,
                    allow_unresolved=allow_unresolved,
                )
                return str(resolved) if resolved is not None else ""

            return cls.SUB_TEMPLATE_REGEX.sub(replace_fn, data)

        elif isinstance(data, dict):
            return {k: cls.resolve_value(v, step_outputs, variables, allow_unresolved) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.resolve_value(item, step_outputs, variables, allow_unresolved) for item in data]
        return data


@dataclass
class WorkflowContext:
    """Manages data flow, step output piping, and cross-application focus state."""

    step_outputs: Dict[str, Any] = field(default_factory=dict)
    active_application: Optional[str] = None
    active_hwnd: Optional[int] = None
    variables: Dict[str, Any] = field(default_factory=dict)

    def record_step_output(self, step_id: str, output: Any) -> None:
        self.step_outputs[step_id] = output

    def get_output(self, step_id: str) -> Any:
        return self.step_outputs.get(step_id)

    def resolve_templates(self, data: Any, allow_unresolved: bool = False) -> Any:
        """Recursively resolve {{step_id.output.key}} or {{variable}} in step arguments."""
        return StrictTemplateResolver.resolve_value(
            data=data,
            step_outputs=self.step_outputs,
            variables=self.variables,
            allow_unresolved=allow_unresolved,
        )


class PersonalWorkflowOrchestrator:
    """Orchestrates multi-application personal assistant workflows across Windows desktop, browser, and filesystem."""

    def __init__(
        self,
        planner: Optional[Planner] = None,
        tool_registry: Optional[ToolRegistry] = None,
        verifier: Optional[Verifier] = None,
        settings: Optional[Settings] = None,
        approval_callback: Optional[Callable[[PlanStep], bool]] = None,
        memory_manager: Optional[MemoryManager] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.tool_registry = tool_registry or registry
        self.verifier = verifier or default_verifier
        self.memory_manager = memory_manager or default_memory_manager
        self.approval_callback = approval_callback
        if planner is not None:
            self.planner = planner
        else:
            try:
                from agent.llm.provider import get_llm_provider
                self.planner = Planner(provider=get_llm_provider(self.settings))
            except Exception:
                from agent.llm.provider import MockLLMProvider
                self.planner = Planner(provider=MockLLMProvider())
        self.context = WorkflowContext()

        # Underpinned strictly by the verified core Agent
        self.agent = Agent(
            planner=self.planner,
            tool_registry=self.tool_registry,
            verifier=self.verifier,
            settings=self.settings,
            approval_callback=self.approval_callback,
            memory_manager=self.memory_manager,
        )

    def execute_workflow(
        self,
        workflow_or_plan: Union[Workflow, Plan],
        initial_variables: Optional[Dict[str, Any]] = None,
        validate: bool = True,
        initial_step_outputs: Optional[Dict[str, Any]] = None,
    ) -> TaskState:
        """Execute a cross-application workflow with dynamic argument resolution and focus synchronization."""
        if isinstance(workflow_or_plan, Workflow):
            plan = workflow_or_plan.plan
            variables = dict(workflow_or_plan.initial_variables)
            variables.update(initial_variables or {})
        else:
            plan = workflow_or_plan
            variables = dict(initial_variables or {})

        if validate:
            WorkflowValidator.validate_plan(
                plan=plan,
                tool_registry=self.tool_registry,
                initial_variables=variables,
            )

        step_outs = dict(initial_step_outputs or {})
        self.context = WorkflowContext(variables=variables, step_outputs=step_outs)

        # Pre-resolve initial static variables into plan step arguments (allowing unresolved step outputs)
        for step in plan.steps:
            step.arguments = self.context.resolve_templates(step.arguments, allow_unresolved=True)

        # Wire pre-security argument resolution and post-verification output recording into Agent FSM
        self.agent._step_argument_resolver = lambda args: self.context.resolve_templates(args, allow_unresolved=False)
        self.agent._step_output_recorder = self.context.record_step_output
        self.agent._pre_dispatch_hook = self._synchronize_focus

        try:
            state = self.agent.run_plan(plan=plan)
            return state
        finally:
            self.agent._step_argument_resolver = None
            self.agent._step_output_recorder = None
            self.agent._pre_dispatch_hook = None

    def _synchronize_focus(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Ensure proper desktop application window focus when transitioning across tools."""
        if tool_name == "computer":
            action = arguments.get("action")
            if action in ("type_text", "set_element_text", "mouse_click", "click_element", "window_focus"):
                hwnd = arguments.get("hwnd") or arguments.get("expected_hwnd")
                if hwnd:
                    comp_tool = self.tool_registry.get("computer")
                    if comp_tool and hasattr(comp_tool, "_bring_window_to_foreground"):
                        try:
                            comp_tool._bring_window_to_foreground(int(hwnd))
                            self.context.active_hwnd = int(hwnd)
                        except Exception as e:
                            logger.warning(f"Failed to focus window {hwnd}: {e}")
        elif tool_name == "browser":
            self.context.active_application = "browser"

    def learn_user_preference(self, key: str, value: str) -> None:
        """Persist user preference to Phase 6 SQLite memory."""
        if self.memory_manager:
            try:
                self.memory_manager.record_user_preference(key, value)
            except Exception as e:
                logger.warning(f"Failed to record preference '{key}': {e}")
