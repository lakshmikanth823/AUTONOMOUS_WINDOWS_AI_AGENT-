"""Unified Perception Layer fusing Win32 window metadata, Native UIA, and Native Windows OCR.

Architecture Principle:
OBSERVE -> FUSE PERCEPTION -> PLAN / RESOLVE TARGET -> SECURITY CHECK -> ACTION -> RE-OBSERVE -> VERIFY
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from agent.tools.ocr import WindowsNativeOCR, OCRResult, OCR_SUCCESS_TEXT_FOUND
from agent.tools.uia import UIAClient, UIElement


class UnifiedTarget(BaseModel):
    """Single unified interactive or semantic target with strict provenance."""

    id: str = Field(description="Deterministic target identifier")
    name: str = Field(description="Normalized target name or label")
    control_type: str = Field(default="Unknown", description="UIA control type or 'Text' for OCR")
    rect: Dict[str, int] = Field(description="Bounding box: left, top, right, bottom, width, height")
    center: Tuple[int, int] = Field(description="Screen coordinates (x, y)")
    enabled: Optional[bool] = Field(default=None, description="Enabled state if known")
    focused: Optional[bool] = Field(default=None, description="Focused state if known")
    hwnd: int = Field(description="Target window handle")
    sources: List[str] = Field(
        default_factory=list,
        description="Provenance sources: ['uia'], ['ocr'], or ['uia', 'ocr']",
    )
    uia_element: Optional[Dict[str, Any]] = Field(
        default=None, description="Underlying UIA element attributes if available"
    )
    ocr_text: Optional[str] = Field(
        default=None, description="Corroborating or recognized OCR text if available"
    )
    ocr_word_count: int = Field(default=0, description="Number of OCR words matching this target")


class PerceivedWindowsState(BaseModel):
    """Unified canonical representation of the perceived Windows desktop state."""

    timestamp: float = Field(default_factory=time.time)
    screen: Dict[str, int] = Field(description="Screen dimensions: width, height")
    cursor: Tuple[int, int] = Field(description="Mouse cursor coordinates (x, y)")
    active_window: Dict[str, Any] = Field(description="Active window metadata: hwnd, title, rect, etc.")
    targets: List[UnifiedTarget] = Field(default_factory=list, description="Deduplicated fused targets")
    uia_count: int = 0
    ocr_word_count: int = 0
    ocr_status: Optional[str] = None
    ocr_performed: bool = False
    contradictions: List[str] = Field(
        default_factory=list, description="Detected perception conflicts or contradictions"
    )


class TargetResolutionResult(BaseModel):
    """Deterministic result of target query resolution."""

    status: str = Field(description="'RESOLVED', 'AMBIGUOUS', 'NOT_FOUND'")
    target: Optional[UnifiedTarget] = None
    candidates: List[UnifiedTarget] = Field(default_factory=list)
    reason: str = ""


def compute_overlap_ratio(rect1: Dict[str, int], rect2: Dict[str, int]) -> float:
    """Compute Intersection-over-Min-Area ratio between two rectangles."""
    l1, t1, r1, b1 = rect1["left"], rect1["top"], rect1["right"], rect1["bottom"]
    l2, t2, r2, b2 = rect2["left"], rect2["top"], rect2["right"], rect2["bottom"]

    inter_left = max(l1, l2)
    inter_top = max(t1, t2)
    inter_right = min(r1, r2)
    inter_bottom = min(b1, b2)

    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0

    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    area1 = max(1, (r1 - l1) * (b1 - t1))
    area2 = max(1, (r2 - l2) * (b2 - t2))
    min_area = min(area1, area2)

    return inter_area / float(min_area)


def fuse_perception(
    screen: Dict[str, int],
    cursor: Tuple[int, int],
    active_window: Dict[str, Any],
    uia_elements: List[Dict[str, Any]],
    ocr_result: Optional[OCRResult] = None,
    overlap_threshold: float = 0.4,
) -> PerceivedWindowsState:
    """Pure fusion function combining Win32, UIA, and OCR observations into unified state."""
    window_hwnd = int(active_window.get("hwnd", 0))
    fused_targets: List[UnifiedTarget] = []
    matched_ocr_word_indices = set()
    contradictions: List[str] = []

    # Flatten OCR words with absolute indices
    ocr_words = []
    ocr_performed = False
    ocr_status = None
    ocr_word_count = 0

    if ocr_result is not None:
        ocr_performed = True
        ocr_status = ocr_result.status
        ocr_word_count = ocr_result.word_count
        word_idx = 0
        for line in ocr_result.lines:
            for w in line.words:
                ocr_words.append((word_idx, w))
                word_idx += 1

    # 1. Process UIA elements and find OCR corroboration
    target_counter = 1
    for el in uia_elements:
        el_name = (el.get("name") or "").strip()
        el_val = (el.get("value") or "").strip()
        el_type = el.get("control_type", "Unknown")
        el_rect = el.get("rect", {})
        el_center = el.get("center", (0, 0))

        # Skip invalid or invisible bounding boxes
        if not el_rect or el_rect.get("width", 0) <= 0 or el_rect.get("height", 0) <= 0:
            continue

        label = el_name or el_val or el_type
        sources = ["uia"]
        corroborating_ocr_text: List[str] = []
        matching_word_count = 0

        # Check OCR overlap if OCR was performed
        # Containers (Window, Pane, Group, TitleBar) must not swallow inner OCR words
        is_container = el_type in ("Window", "Pane", "Group", "TitleBar")
        if ocr_performed and ocr_words and not is_container:
            for w_idx, ocr_w in ocr_words:
                ratio = compute_overlap_ratio(el_rect, ocr_w.rect)
                text_match = (
                    ocr_w.text.lower() in label.lower()
                    or label.lower() in ocr_w.text.lower()
                )
                if ratio >= overlap_threshold and text_match:
                    matched_ocr_word_indices.add(w_idx)
                    corroborating_ocr_text.append(ocr_w.text)
                    matching_word_count += 1

        if corroborating_ocr_text:
            sources.append("ocr")

        tgt_id = f"tgt_{target_counter:03d}_{label.lower()[:15].replace(' ', '_')}"
        target_counter += 1

        fused_targets.append(
            UnifiedTarget(
                id=tgt_id,
                name=label,
                control_type=el_type,
                rect=el_rect,
                center=el_center,
                enabled=el.get("enabled"),
                focused=el.get("focused"),
                hwnd=window_hwnd,
                sources=sources,
                uia_element=el,
                ocr_text=" ".join(corroborating_ocr_text) if corroborating_ocr_text else None,
                ocr_word_count=matching_word_count,
            )
        )

    # 2. Process unmatched OCR words as OCR-only targets (canvas, custom rendering)
    if ocr_performed and ocr_words:
        unmatched_words = [w for w_idx, w in ocr_words if w_idx not in matched_ocr_word_indices]
        for w in unmatched_words:
            w_text = w.text.strip()
            if not w_text:
                continue

            tgt_id = f"tgt_{target_counter:03d}_{w_text.lower()[:15].replace(' ', '_')}"
            target_counter += 1

            fused_targets.append(
                UnifiedTarget(
                    id=tgt_id,
                    name=w_text,
                    control_type="Text",
                    rect=w.rect,
                    center=w.center,
                    enabled=True,
                    focused=False,
                    hwnd=window_hwnd,
                    sources=["ocr"],
                    uia_element=None,
                    ocr_text=w_text,
                    ocr_word_count=1,
                )
            )

    return PerceivedWindowsState(
        timestamp=time.time(),
        screen=screen,
        cursor=cursor,
        active_window=active_window,
        targets=fused_targets,
        uia_count=len(uia_elements),
        ocr_word_count=ocr_word_count,
        ocr_status=ocr_status,
        ocr_performed=ocr_performed,
        contradictions=contradictions,
    )


def resolve_target(
    targets: List[UnifiedTarget],
    query: str,
    control_type_filter: Optional[str] = None,
) -> TargetResolutionResult:
    """Deterministically resolve a semantic query to a single target with priority and ambiguity gates."""
    q = query.strip().lower()
    if not q:
        return TargetResolutionResult(status="NOT_FOUND", reason="Empty target query.")

    candidates_p1 = []  # Exact UIA + OCR corroborated match
    candidates_p2 = []  # Exact UIA match
    candidates_p3 = []  # Exact OCR match
    candidates_p4 = []  # Partial UIA match
    candidates_p5 = []  # Partial OCR match

    for tgt_raw in targets:
        tgt = UnifiedTarget(**tgt_raw) if isinstance(tgt_raw, dict) else tgt_raw

        # Control type filtering if requested
        if control_type_filter:
            if tgt.control_type.lower() != control_type_filter.strip().lower():
                continue

        name_lower = tgt.name.lower()
        ocr_lower = (tgt.ocr_text or "").lower()
        is_corroborated = "uia" in tgt.sources and "ocr" in tgt.sources
        is_uia = "uia" in tgt.sources and "ocr" not in tgt.sources
        is_ocr = "ocr" in tgt.sources and "uia" not in tgt.sources

        # Check exact equality
        if name_lower == q or ocr_lower == q or tgt.control_type.lower() == q:
            if is_corroborated:
                candidates_p1.append(tgt)
            elif is_uia:
                candidates_p2.append(tgt)
            elif is_ocr:
                candidates_p3.append(tgt)
            continue

        # Check substring containment
        if q in name_lower or q in ocr_lower:
            if is_corroborated or is_uia:
                candidates_p4.append(tgt)
            elif is_ocr:
                candidates_p5.append(tgt)

    # Deterministic priority hierarchy
    ordered_tiers = [
        (candidates_p1, "Exact UIA+OCR corroborated match"),
        (candidates_p2, "Exact UIA match"),
        (candidates_p3, "Exact OCR match"),
        (candidates_p4, "Partial UIA match"),
        (candidates_p5, "Partial OCR match"),
    ]

    for tier_candidates, tier_name in ordered_tiers:
        if not tier_candidates:
            continue

        if len(tier_candidates) == 1:
            return TargetResolutionResult(
                status="RESOLVED",
                target=tier_candidates[0],
                candidates=tier_candidates,
                reason=f"Resolved via {tier_name}.",
            )

        # Ambiguity Gate: Multiple candidates tie in the same highest priority tier
        cand_summary = ", ".join(f"'{c.name}'[{c.control_type}] at {c.center}" for c in tier_candidates[:5])
        return TargetResolutionResult(
            status="AMBIGUOUS",
            target=None,
            candidates=tier_candidates,
            reason=f"Ambiguous target: found {len(tier_candidates)} candidates tying for {tier_name} ({cand_summary}). Action aborted.",
        )

    return TargetResolutionResult(
        status="NOT_FOUND",
        target=None,
        candidates=[],
        reason=f"Target '{query}' not found in perception targets.",
    )


class PerceptionEngine:
    """Coordinates selective Win32, UIA, and Native OCR capture and fusion."""

    def __init__(
        self,
        uia_client: Optional[UIAClient] = None,
        ocr_engine: Optional[WindowsNativeOCR] = None,
    ) -> None:
        self.uia = uia_client or UIAClient()
        self.ocr = ocr_engine or WindowsNativeOCR()

    def observe(
        self,
        screen_size: Tuple[int, int],
        cursor_pos: Tuple[int, int],
        active_window_info: Dict[str, Any],
        ocr_mode: str = "off",  # "off", "auto", "region", "screen"
        max_depth: int = 5,
        max_elements: int = 100,
        control_type: Optional[str] = None,
        hwnd: Optional[int] = None,
    ) -> PerceivedWindowsState:
        """Perform unified perception observation with selective OCR."""
        target_hwnd = hwnd or active_window_info.get("hwnd", 0)

        # 1. Win32 screen and cursor
        screen = {"width": screen_size[0], "height": screen_size[1]}

        # 2. UIA element enumeration
        uia_res = self.uia.get_active_window_elements(
            hwnd=target_hwnd,
            max_depth=max_depth,
            max_elements=max_elements,
            control_type_filter=control_type,
        )
        elements = uia_res.get("elements", [])

        # 3. Selective OCR execution
        ocr_result: Optional[OCRResult] = None

        if ocr_mode == "screen":
            ocr_result = self.ocr.recognize_screen(hwnd=target_hwnd)
        elif ocr_mode == "region":
            rect = dict(active_window_info.get("rect", {}))
            if target_hwnd:
                import ctypes
                from ctypes import wintypes
                rc = wintypes.RECT()
                if ctypes.windll.user32.GetWindowRect(target_hwnd, ctypes.byref(rc)):
                    rect = {
                        "left": rc.left,
                        "top": rc.top,
                        "right": rc.right,
                        "bottom": rc.bottom,
                        "width": max(0, rc.right - rc.left),
                        "height": max(0, rc.bottom - rc.top),
                    }
            rx = max(0, int(rect.get("left", 0)))
            ry = max(0, int(rect.get("top", 0)))
            rw = max(1, int(rect.get("width", screen_size[0])))
            rh = max(1, int(rect.get("height", screen_size[1])))
            ocr_result = self.ocr.recognize_region(x=rx, y=ry, width=rw, height=rh, hwnd=target_hwnd)
        elif ocr_mode == "auto":
            # Auto triggers OCR only if UIA returned 0 elements (e.g. canvas or game)
            if len(elements) == 0:
                rect = active_window_info.get("rect", {})
                rx = max(0, int(rect.get("left", 0)))
                ry = max(0, int(rect.get("top", 0)))
                rw = max(1, int(rect.get("width", 600)))
                rh = max(1, int(rect.get("height", 400)))
                ocr_result = self.ocr.recognize_region(x=rx, y=ry, width=rw, height=rh, hwnd=target_hwnd)

        # 4. Pure perception fusion
        return fuse_perception(
            screen=screen,
            cursor=cursor_pos,
            active_window=active_window_info,
            uia_elements=elements,
            ocr_result=ocr_result,
        )
