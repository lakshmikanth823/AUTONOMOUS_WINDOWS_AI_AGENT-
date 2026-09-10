"""Working memory tracking in-flight execution context, scratchpad, and perception reconciliation."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WorkingMemoryHypothesis(BaseModel):
    """An unverified claim retrieved from long-term memory awaiting live corroboration."""

    memory_id: str
    category: str
    claim: str
    expected_key: Optional[str] = None
    expected_value: Optional[str] = None
    verified: Optional[bool] = None  # None = pending, True = verified, False = refuted
    refutation_reason: Optional[str] = None
    created_at: float = Field(default_factory=time.time)


class WorkingMemory(BaseModel):
    """In-flight working memory for the active agent execution loop."""

    active_window: Optional[Dict[str, Any]] = None
    active_browser_tab: Optional[Dict[str, Any]] = None
    scratchpad: Dict[str, Any] = Field(default_factory=dict)
    hypotheses: List[WorkingMemoryHypothesis] = Field(default_factory=list)
    verified_facts: Dict[str, Any] = Field(default_factory=dict)
    contradicted_memories: List[Dict[str, Any]] = Field(default_factory=list)

    def update_active_window(
        self,
        hwnd: int,
        title: str,
        pid: Optional[int] = None,
        process_name: Optional[str] = None,
    ) -> None:
        """Update active foreground window context in working memory."""
        self.active_window = {
            "hwnd": hwnd,
            "title": title,
            "pid": pid,
            "process_name": process_name,
            "updated_at": time.time(),
        }

    def update_active_tab(self, tab_id: int, url: str, title: str) -> None:
        """Update active browser tab context in working memory."""
        self.active_browser_tab = {
            "tab_id": tab_id,
            "url": url,
            "title": title,
            "updated_at": time.time(),
        }

    def set_scratchpad(self, key: str, value: Any) -> None:
        """Store an ephemeral intermediate key-value discovered during the task."""
        self.scratchpad[key] = value

    def get_scratchpad(self, key: str, default: Any = None) -> Any:
        """Retrieve an ephemeral scratchpad value."""
        return self.scratchpad.get(key, default)

    def add_hypothesis(
        self,
        memory_id: str,
        category: str,
        claim: str,
        expected_key: Optional[str] = None,
        expected_value: Optional[str] = None,
    ) -> WorkingMemoryHypothesis:
        """Record an unverified hypothesis from long-term memory for live corroboration."""
        hyp = WorkingMemoryHypothesis(
            memory_id=memory_id,
            category=category,
            claim=claim,
            expected_key=expected_key,
            expected_value=expected_value,
        )
        self.hypotheses.append(hyp)
        return hyp

    def reconcile_observation(self, live_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Reconcile active memory hypotheses against live perception.
        
        CRITICAL SAFETY RULE: Perception unconditionally overrides memory.
        If a hypothesis is refuted by live observation, it is marked refuted
        and queued for long-term memory invalidation or superseding.
        """
        corroborated: List[str] = []
        refuted: List[Dict[str, Any]] = []

        # Extract observable reality
        obs_title = (
            live_obs.get("window", {}).get("title")
            or live_obs.get("title")
            or (self.active_window.get("title") if self.active_window else "")
        )
        obs_hwnd = (
            live_obs.get("window", {}).get("hwnd")
            or live_obs.get("hwnd")
            or (self.active_window.get("hwnd") if self.active_window else None)
        )
        obs_elements = [
            t.get("element_name", "") if isinstance(t, dict) else str(t)
            for t in live_obs.get("targets", []) or live_obs.get("elements", [])
        ]

        for hyp in self.hypotheses:
            if hyp.verified is not None:
                continue  # already adjudicated

            # 1. Window title hypothesis
            if hyp.expected_key == "window_title" and hyp.expected_value:
                exp = hyp.expected_value.strip().lower()
                actual = (obs_title or "").strip().lower()
                if exp in actual or actual in exp:
                    hyp.verified = True
                    corroborated.append(hyp.memory_id)
                    self.verified_facts[hyp.expected_key] = obs_title
                else:
                    hyp.verified = False
                    reason = f"Perception contradiction: memory asserted window title '{hyp.expected_value}', but observed '{obs_title}'."
                    hyp.refutation_reason = reason
                    contradiction_entry = {
                        "memory_id": hyp.memory_id,
                        "claim": hyp.claim,
                        "expected": hyp.expected_value,
                        "observed": obs_title,
                        "reason": reason,
                    }
                    refuted.append(contradiction_entry)
                    self.contradicted_memories.append(contradiction_entry)

            # 2. Window HWND hypothesis
            elif hyp.expected_key == "hwnd" and hyp.expected_value:
                try:
                    exp_h = int(hyp.expected_value)
                    if obs_hwnd and int(obs_hwnd) == exp_h:
                        hyp.verified = True
                        corroborated.append(hyp.memory_id)
                    elif obs_hwnd and int(obs_hwnd) != exp_h:
                        hyp.verified = False
                        reason = f"Perception contradiction: memory asserted HWND {exp_h}, but active window is {obs_hwnd}."
                        hyp.refutation_reason = reason
                        contradiction_entry = {
                            "memory_id": hyp.memory_id,
                            "claim": hyp.claim,
                            "expected": exp_h,
                            "observed": obs_hwnd,
                            "reason": reason,
                        }
                        refuted.append(contradiction_entry)
                        self.contradicted_memories.append(contradiction_entry)
                except ValueError:
                    pass

            # 3. Target element hypothesis
            elif hyp.expected_key == "element" and hyp.expected_value:
                target_name = hyp.expected_value.strip().lower()
                matched = any(target_name in el.lower() for el in obs_elements)
                if matched:
                    hyp.verified = True
                    corroborated.append(hyp.memory_id)
                elif obs_elements:  # only refute if observation was successfully captured
                    hyp.verified = False
                    reason = f"Perception contradiction: element '{hyp.expected_value}' not found among {len(obs_elements)} observed controls."
                    hyp.refutation_reason = reason
                    contradiction_entry = {
                        "memory_id": hyp.memory_id,
                        "claim": hyp.claim,
                        "expected": hyp.expected_value,
                        "observed": obs_elements[:5],
                        "reason": reason,
                    }
                    refuted.append(contradiction_entry)
                    self.contradicted_memories.append(contradiction_entry)

        return {
            "corroborated_ids": corroborated,
            "refuted_records": refuted,
            "total_hypotheses": len(self.hypotheses),
        }

    def clear(self) -> None:
        """Reset working memory context for a new task execution."""
        self.active_window = None
        self.active_browser_tab = None
        self.scratchpad.clear()
        self.hypotheses.clear()
        self.verified_facts.clear()
        self.contradicted_memories.clear()
