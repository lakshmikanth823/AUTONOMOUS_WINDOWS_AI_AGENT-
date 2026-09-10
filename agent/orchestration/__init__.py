"""Multi-application autonomous workflow orchestration for the Personal Windows Agent."""

from agent.orchestration.models import (
    TemplateResolutionError,
    Workflow,
    WorkflowValidationError,
    WorkflowValidator,
)
from agent.orchestration.workflow import (
    PersonalWorkflowOrchestrator,
    StrictTemplateResolver,
    WorkflowContext,
)
from agent.orchestration.archetypes import (
    ResearchAndSynthesizeWorkflow,
    WorkspaceSetupWorkflow,
    FileOrganizationWorkflow,
)

__all__ = [
    "PersonalWorkflowOrchestrator",
    "StrictTemplateResolver",
    "WorkflowContext",
    "Workflow",
    "WorkflowValidator",
    "WorkflowValidationError",
    "TemplateResolutionError",
    "ResearchAndSynthesizeWorkflow",
    "WorkspaceSetupWorkflow",
    "FileOrganizationWorkflow",
]
