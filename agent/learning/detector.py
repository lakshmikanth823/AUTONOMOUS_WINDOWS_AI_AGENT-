"""Deterministic pattern detection and confidence calculation from verified activity."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from agent.learning.models import LearningRecord, LearningStatus, PatternType
from agent.security.redactor import SecretRedactor

logger = logging.getLogger(__name__)


class PatternDetector:
    """Deterministic, code-free pattern detector evaluating historical task activity."""

    def __init__(self, min_occurrence_threshold: int = 1) -> None:
        self.min_occurrence_threshold = max(1, min_occurrence_threshold)

    @staticmethod
    def calculate_confidence(
        occurrence_count: int,
        success_rate: float = 1.0,
        recency_hours: float = 0.0,
        rejection_count: int = 0,
    ) -> float:
        """Calculate deterministic bounded confidence score in [0.0, 1.0]."""
        if occurrence_count < 1:
            return 0.0

        # Asymptotic frequency score: 2->0.50, 3->0.60, 5->0.71, 10->0.83, 20->0.91
        base = 1.0 - (1.0 / (1.0 + 0.5 * occurrence_count))

        # Recency decay over 30 days (720 hours)
        recency_factor = max(0.5, 1.0 - (recency_hours / 720.0))

        # Rejection penalty
        rejection_penalty = 0.5 ** rejection_count

        conf = base * max(0.0, min(1.0, success_rate)) * recency_factor * rejection_penalty
        return round(max(0.0, min(1.0, conf)), 4)

    def detect_action_sequences(
        self,
        task_executions: List[Dict[str, Any]],
        rejection_counts: Optional[Dict[str, int]] = None,
    ) -> List[LearningRecord]:
        """Identify repeated sequences of tool actions across multiple verified tasks."""
        rejection_counts = rejection_counts or {}
        sequence_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for task in task_executions:
            # Only consider verified / successful tasks
            if not task.get("success", False) and task.get("status") not in ("COMPLETED", "completed"):
                continue

            steps = task.get("steps", [])
            if len(steps) < 2:
                continue

            # Build canonical signature of sequential tool actions: e.g. "browser:launch->filesystem:create_file->computer:set_element_text"
            sig_parts = []
            canonical_steps = []
            for s in steps:
                tool = s.get("tool_required") or s.get("tool_name") or "unknown"
                args = s.get("arguments", {})
                action = args.get("action") or s.get("action") or "default"
                sig_parts.append(f"{tool}:{action}")
                canonical_steps.append({"tool": tool, "action": action})

            sig = "->".join(sig_parts)
            sequence_map[sig].append(task)

        records: List[LearningRecord] = []
        for sig, matching_tasks in sequence_map.items():
            count = len(matching_tasks)
            if count < self.min_occurrence_threshold:
                continue

            # Generate deterministic learning ID from signature
            sig_hash = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:12]
            learning_id = f"lrn_seq_{sig_hash}"
            rej = rejection_counts.get(learning_id, 0)
            conf = self.calculate_confidence(count, success_rate=1.0, rejection_count=rej)

            first_seen = min(t.get("created_at") or datetime.now(timezone.utc).isoformat() for t in matching_tasks)
            last_seen = max(t.get("created_at") or datetime.now(timezone.utc).isoformat() for t in matching_tasks)

            rec = LearningRecord(
                learning_id=learning_id,
                pattern_type=PatternType.ACTION_SEQUENCE,
                description=f"Repeated action sequence executed {count} times: {sig}",
                evidence={
                    "sequence_signature": sig,
                    "task_count": count,
                    "sample_task_ids": [t.get("task_id") for t in matching_tasks[:5] if t.get("task_id")],
                    "steps": matching_tasks[0].get("steps", [])[:10],
                },
                confidence=conf,
                occurrence_count=count,
                first_seen=first_seen,
                last_seen=last_seen,
                source_task_id=matching_tasks[-1].get("task_id"),
                status=LearningStatus.CONFIRMED if conf >= 0.6 else LearningStatus.CANDIDATE,
                metadata={"signature": sig},
            )
            records.append(rec)

        return records

    def detect_app_launches(
        self,
        launch_events: List[Dict[str, Any]],
        rejection_counts: Optional[Dict[str, int]] = None,
    ) -> List[LearningRecord]:
        """Detect applications launched repeatedly or clustered around specific times."""
        rejection_counts = rejection_counts or {}
        app_counts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for ev in launch_events:
            app_name = ev.get("app_name") or ev.get("command") or ""
            if not app_name:
                continue
            norm_name = app_name.strip().lower()
            app_counts[norm_name].append(ev)

        records: List[LearningRecord] = []
        for norm_app, events in app_counts.items():
            count = len(events)
            if count < self.min_occurrence_threshold:
                continue

            app_hash = hashlib.sha256(norm_app.encode("utf-8")).hexdigest()[:12]
            learning_id = f"lrn_app_{app_hash}"
            rej = rejection_counts.get(learning_id, 0)
            conf = self.calculate_confidence(count, success_rate=1.0, rejection_count=rej)

            # Analyze time of day clustering if timestamps exist
            times_of_day = []
            for ev in events:
                ts_str = ev.get("timestamp") or ev.get("created_at")
                if ts_str:
                    try:
                        dt = datetime.fromisoformat(ts_str)
                        times_of_day.append(f"{dt.hour:02d}:00")
                    except Exception:
                        pass

            common_hour = Counter(times_of_day).most_common(1)[0][0] if times_of_day else "09:00"

            rec = LearningRecord(
                learning_id=learning_id,
                pattern_type=PatternType.APP_LAUNCH_TIME,
                description=f"Application '{norm_app}' repeatedly launched ({count} times, frequently around {common_hour})",
                evidence={
                    "app_name": norm_app,
                    "launch_count": count,
                    "common_hour": common_hour,
                    "sample_events": events[:5],
                },
                confidence=conf,
                occurrence_count=count,
                first_seen=min((e.get("timestamp") or datetime.now(timezone.utc).isoformat()) for e in events),
                last_seen=max((e.get("timestamp") or datetime.now(timezone.utc).isoformat()) for e in events),
                status=LearningStatus.CONFIRMED if conf >= 0.6 else LearningStatus.CANDIDATE,
                metadata={"app_name": norm_app, "suggested_time": common_hour},
            )
            records.append(rec)

        return records

    def detect_repeated_workflows(
        self,
        workflow_runs: List[Dict[str, Any]],
        rejection_counts: Optional[Dict[str, int]] = None,
    ) -> List[LearningRecord]:
        """Detect multi-app workflows that are executed repeatedly."""
        rejection_counts = rejection_counts or {}
        wf_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for run in workflow_runs:
            wf_id = run.get("workflow_id") or run.get("goal") or ""
            if not wf_id:
                continue
            wf_map[wf_id].append(run)

        records: List[LearningRecord] = []
        for wf_id, runs in wf_map.items():
            count = len(runs)
            if count < self.min_occurrence_threshold:
                continue

            successes = sum(1 for r in runs if r.get("status") in ("COMPLETED", "completed", TaskStateEnum_COMPLETED := "COMPLETED"))
            rate = successes / count if count > 0 else 0.0

            wf_hash = hashlib.sha256(wf_id.encode("utf-8")).hexdigest()[:12]
            learning_id = f"lrn_wf_{wf_hash}"
            rej = rejection_counts.get(learning_id, 0)
            conf = self.calculate_confidence(count, success_rate=rate, rejection_count=rej)

            rec = LearningRecord(
                learning_id=learning_id,
                pattern_type=PatternType.REPEATED_WORKFLOW,
                description=f"Workflow '{wf_id}' executed repeatedly ({count} runs, {successes} successful)",
                evidence={
                    "workflow_id": wf_id,
                    "total_runs": count,
                    "success_count": successes,
                    "success_rate": round(rate, 2),
                },
                confidence=conf,
                occurrence_count=count,
                source_workflow_id=wf_id,
                status=LearningStatus.CONFIRMED if conf >= 0.6 else LearningStatus.CANDIDATE,
                metadata={"workflow_id": wf_id},
            )
            records.append(rec)

        return records

    def detect_failure_recoveries(
        self,
        recovery_events: List[Dict[str, Any]],
        rejection_counts: Optional[Dict[str, int]] = None,
    ) -> List[LearningRecord]:
        """Identify recurring errors that consistently resolve through a specific recovery step."""
        rejection_counts = rejection_counts or {}
        strat_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for ev in recovery_events:
            err_type = ev.get("error_type") or ev.get("initial_error") or ""
            recovery_action = ev.get("successful_recovery_action") or ev.get("recovery_action") or ""
            if not err_type or not recovery_action:
                continue

            key = f"{err_type}->{recovery_action}"
            strat_map[key].append(ev)

        records: List[LearningRecord] = []
        for key, events in strat_map.items():
            count = len(events)
            if count < self.min_occurrence_threshold:
                continue

            parts = key.split("->", 1)
            err_type, recovery_action = parts[0], parts[1]
            key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
            learning_id = f"lrn_rec_{key_hash}"
            rej = rejection_counts.get(learning_id, 0)
            conf = self.calculate_confidence(count, success_rate=1.0, rejection_count=rej)

            rec = LearningRecord(
                learning_id=learning_id,
                pattern_type=PatternType.RECURRING_FAILURE_RECOVERY,
                description=f"Failure '{err_type}' repeatedly resolved via recovery procedure '{recovery_action}' ({count} times)",
                evidence={
                    "error_type": err_type,
                    "recovery_action": recovery_action,
                    "success_count": count,
                },
                confidence=conf,
                occurrence_count=count,
                status=LearningStatus.CONFIRMED if conf >= 0.6 else LearningStatus.CANDIDATE,
                metadata={"error_type": err_type, "recovery_action": recovery_action},
            )
            records.append(rec)

        return records
