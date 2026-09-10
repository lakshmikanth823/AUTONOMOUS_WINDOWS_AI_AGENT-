"""Dependency graph representations, cycle detection, topological sorting, and ready subgoal resolution."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Set

from agent.exceptions import PlanValidationError
from agent.planning.models import Subgoal, SubgoalStatus


class DependencyGraph:
    """Lightweight directed acyclic graph (DAG) governing subgoal execution order."""

    def __init__(self, subgoals: List[Subgoal]) -> None:
        self.subgoals_map: Dict[str, Subgoal] = {}
        # adj[u] = list of subgoals that depend on u (downstream)
        self.downstream: Dict[str, List[str]] = {}
        # in_degrees: number of unmet prerequisites for each subgoal
        self.in_degree: Dict[str, int] = {}
        
        self._build_and_validate(subgoals)

    def _build_and_validate(self, subgoals: List[Subgoal]) -> None:
        if not subgoals:
            raise PlanValidationError("Plan contains zero subgoals.")

        # 1. Check duplicate IDs and populate map
        for sg in subgoals:
            sid = sg.subgoal_id.strip() if sg.subgoal_id else ""
            if not sid:
                raise PlanValidationError("Subgoal has an empty or whitespace subgoal_id.")
            if sid in self.subgoals_map:
                raise PlanValidationError(f"Duplicate subgoal ID detected: '{sid}'.")
            self.subgoals_map[sid] = sg
            self.downstream[sid] = []
            self.in_degree[sid] = 0

        # 2. Check dependencies and build adjacency
        for sid, sg in self.subgoals_map.items():
            for dep in sg.dependencies:
                dep_id = dep.strip()
                if not dep_id:
                    raise PlanValidationError(f"Subgoal '{sid}' has an empty dependency reference.")
                if dep_id == sid:
                    raise PlanValidationError(f"Circular self-dependency detected: subgoal '{sid}' depends on itself.")
                if dep_id not in self.subgoals_map:
                    raise PlanValidationError(
                        f"Missing dependency reference: subgoal '{sid}' depends on non-existent subgoal '{dep_id}'."
                    )
                self.downstream[dep_id].append(sid)
                self.in_degree[sid] += 1

        # 3. Detect cycles via Kahn's algorithm
        self._detect_cycles()

    def _detect_cycles(self) -> None:
        """Enforce strict acyclicity using Kahn's algorithm."""
        in_deg_copy = dict(self.in_degree)
        queue = deque([sid for sid, deg in in_deg_copy.items() if deg == 0])
        visited_count = 0

        while queue:
            node = queue.popleft()
            visited_count += 1
            for nxt in self.downstream[node]:
                in_deg_copy[nxt] -= 1
                if in_deg_copy[nxt] == 0:
                    queue.append(nxt)

        if visited_count < len(self.subgoals_map):
            # There is at least one cycle
            unresolved = [sid for sid, deg in in_deg_copy.items() if deg > 0]
            raise PlanValidationError(
                f"Circular dependency detected in plan graph among subgoals: {unresolved}."
            )

    def get_topological_order(self) -> List[str]:
        """Return a valid linear ordering of subgoal IDs respecting all dependencies."""
        in_deg = dict(self.in_degree)
        queue = deque([sid for sid, deg in in_deg.items() if deg == 0])
        order: List[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for nxt in self.downstream[node]:
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    queue.append(nxt)
        return order

    def get_ready_subgoals(self) -> List[Subgoal]:
        """Return subgoals whose prerequisites are fully COMPLETED and status is PENDING or READY."""
        ready: List[Subgoal] = []
        for sid, sg in self.subgoals_map.items():
            if sg.status in (SubgoalStatus.PENDING, SubgoalStatus.READY):
                # Check if all dependencies are COMPLETED
                deps_satisfied = True
                for dep in sg.dependencies:
                    parent_sg = self.subgoals_map.get(dep)
                    if not parent_sg or parent_sg.status != SubgoalStatus.COMPLETED:
                        deps_satisfied = False
                        break
                if deps_satisfied:
                    ready.append(sg)
        # Sort ready by priority descending (higher priority first)
        ready.sort(key=lambda s: s.priority, reverse=True)
        return ready

    def mark_subgoal_failed(self, subgoal_id: str) -> List[str]:
        """Mark a subgoal FAILED and cascade BLOCKED status to all downstream dependent subgoals."""
        sg = self.subgoals_map.get(subgoal_id)
        if sg:
            sg.status = SubgoalStatus.FAILED

        blocked_ids: List[str] = []
        queue = deque(self.downstream.get(subgoal_id, []))
        visited: Set[str] = set()

        while queue:
            downstream_id = queue.popleft()
            if downstream_id in visited:
                continue
            visited.add(downstream_id)
            down_sg = self.subgoals_map.get(downstream_id)
            if down_sg and not down_sg.is_terminal():
                down_sg.status = SubgoalStatus.BLOCKED
                blocked_ids.append(downstream_id)
            for nxt in self.downstream.get(downstream_id, []):
                queue.append(nxt)

        return blocked_ids

    def is_complete(self) -> bool:
        """Check if all subgoals in the graph have reached a terminal status."""
        return all(sg.is_terminal() for sg in self.subgoals_map.values())

    def all_successful(self) -> bool:
        """Check if every subgoal completed successfully."""
        return all(sg.status == SubgoalStatus.COMPLETED for sg in self.subgoals_map.values())
