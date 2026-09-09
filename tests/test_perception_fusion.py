"""Comprehensive test suite for Phase 3C: Perception Fusion, Target Disambiguation, and Safety."""

import pytest
from typing import Any, Dict, List

from agent.config.permissions import PermissionLevel
from agent.core.verifier import VerificationStatus, default_verifier
from agent.security.policy import SecurityPolicy
from agent.tools.base import ToolResult
from agent.tools.computer import ComputerTool
from agent.tools.ocr import OCRLine, OCRResult, OCRWord, OCR_SUCCESS_TEXT_FOUND
from agent.tools.perception import (
    PerceivedWindowsState,
    UnifiedTarget,
    compute_overlap_ratio,
    fuse_perception,
    resolve_target,
)


class TestPerceptionFusionPure:
    """Test pure spatial and semantic fusion logic without hardware/display dependencies."""

    @pytest.fixture
    def mock_screen(self) -> Dict[str, int]:
        return {"width": 1920, "height": 1080}

    @pytest.fixture
    def mock_cursor(self) -> tuple[int, int]:
        return (100, 200)

    @pytest.fixture
    def mock_window(self) -> Dict[str, Any]:
        return {
            "hwnd": 12345,
            "title": "Test App Window",
            "process_id": 9999,
            "process_name": "testapp.exe",
            "rect": {"left": 100, "top": 100, "right": 900, "bottom": 700, "width": 800, "height": 600},
        }

    def test_spatial_overlap_calculation(self) -> None:
        """Test intersection-over-min-area calculation across various spatial configurations."""
        # Exact identical boxes
        b1 = {"left": 50, "top": 50, "right": 150, "bottom": 100, "width": 100, "height": 50}
        assert compute_overlap_ratio(b1, b1) == 1.0

        # Completely disjoint boxes
        b2 = {"left": 200, "top": 200, "right": 300, "bottom": 250, "width": 100, "height": 50}
        assert compute_overlap_ratio(b1, b2) == 0.0

        # Enclosing / Sub-box (b3 inside b1)
        b3 = {"left": 60, "top": 60, "right": 100, "bottom": 90, "width": 40, "height": 30}
        # Area of b3 is 40*30 = 1200. Inter area is 1200. Ratio = 1200 / min(5000, 1200) = 1.0
        assert compute_overlap_ratio(b1, b3) == 1.0

        # Partial overlap (50% of b4 inside b1)
        b4 = {"left": 100, "top": 50, "right": 200, "bottom": 100, "width": 100, "height": 50}
        # Inter: [100, 50, 150, 100] -> width 50, height 50 = 2500. Area1=5000, Area4=5000.
        assert compute_overlap_ratio(b1, b4) == 0.5

    def test_uia_only_target(self, mock_screen, mock_cursor, mock_window) -> None:
        """UIA element with no OCR corroboration yields single target with source=['uia']."""
        uia_elems = [
            {
                "name": "OK",
                "control_type": "Button",
                "rect": {"left": 200, "top": 300, "right": 280, "bottom": 340, "width": 80, "height": 40},
                "center": (240, 320),
                "enabled": True,
                "focused": False,
            }
        ]
        state = fuse_perception(mock_screen, mock_cursor, mock_window, uia_elems, ocr_result=None)
        assert len(state.targets) == 1
        tgt = state.targets[0]
        assert tgt.name == "OK"
        assert tgt.control_type == "Button"
        assert tgt.sources == ["uia"]
        assert tgt.hwnd == 12345
        assert tgt.center == (240, 320)
        assert tgt.ocr_text is None
        assert state.ocr_performed is False

    def test_ocr_only_target(self, mock_screen, mock_cursor, mock_window) -> None:
        """Canvas text recognized by OCR but absent from UIA yields target with source=['ocr']."""
        ocr_res = OCRResult(
            status=OCR_SUCCESS_TEXT_FOUND,
            text="Unrendered Canvas Label",
            word_count=3,
            line_count=1,
            lines=[
                OCRLine(
                    text="Unrendered Canvas Label",
                    rect={"left": 400, "top": 200, "right": 650, "bottom": 240, "width": 250, "height": 40},
                    center=(525, 220),
                    words=[
                        OCRWord(
                            text="Unrendered",
                            rect={"left": 400, "top": 200, "right": 480, "bottom": 240, "width": 80, "height": 40},
                            center=(440, 220),
                        ),
                        OCRWord(
                            text="Canvas",
                            rect={"left": 490, "top": 200, "right": 560, "bottom": 240, "width": 70, "height": 40},
                            center=(525, 220),
                        ),
                        OCRWord(
                            text="Label",
                            rect={"left": 570, "top": 200, "right": 650, "bottom": 240, "width": 80, "height": 40},
                            center=(610, 220),
                        ),
                    ],
                )
            ],
        )
        state = fuse_perception(mock_screen, mock_cursor, mock_window, uia_elements=[], ocr_result=ocr_res)
        assert state.ocr_performed is True
        assert len(state.targets) == 3
        assert state.targets[0].name == "Unrendered"
        assert state.targets[0].sources == ["ocr"]
        assert state.targets[0].control_type == "Text"

    def test_corroborated_target_merging(self, mock_screen, mock_cursor, mock_window) -> None:
        """Target seen by both UIA and OCR merges into one target with sources=['uia', 'ocr']."""
        uia_elems = [
            {
                "name": "Submit",
                "control_type": "Button",
                "rect": {"left": 300, "top": 400, "right": 420, "bottom": 450, "width": 120, "height": 50},
                "center": (360, 425),
                "enabled": True,
                "focused": True,
            }
        ]
        ocr_res = OCRResult(
            status=OCR_SUCCESS_TEXT_FOUND,
            text="Submit",
            word_count=1,
            line_count=1,
            lines=[
                OCRLine(
                    text="Submit",
                    rect={"left": 310, "top": 410, "right": 400, "bottom": 440, "width": 90, "height": 30},
                    center=(355, 425),
                    words=[
                        OCRWord(
                            text="Submit",
                            rect={"left": 310, "top": 410, "right": 400, "bottom": 440, "width": 90, "height": 30},
                            center=(355, 425),
                        )
                    ],
                )
            ],
        )
        state = fuse_perception(mock_screen, mock_cursor, mock_window, uia_elems, ocr_res)
        # Must merge into ONE target, not two!
        assert len(state.targets) == 1
        tgt = state.targets[0]
        assert tgt.name == "Submit"
        assert tgt.control_type == "Button"
        assert set(tgt.sources) == {"uia", "ocr"}
        assert tgt.ocr_text == "Submit"
        assert tgt.ocr_word_count == 1
        assert tgt.center == (360, 425)

    def test_case_a_overlapping_and_matching_text(self, mock_screen, mock_cursor, mock_window) -> None:
        """Case A: UIA 'Save' + OCR 'Save' + overlapping -> 1 target, sources=['uia', 'ocr']."""
        uia_elems = [
            {"name": "Save", "control_type": "Button", "rect": {"left": 100, "top": 100, "right": 180, "bottom": 140, "width": 80, "height": 40}, "center": (140, 120)}
        ]
        ocr_res = OCRResult(
            status=OCR_SUCCESS_TEXT_FOUND, text="Save", word_count=1, line_count=1,
            lines=[OCRLine(text="Save", rect={"left": 105, "top": 105, "right": 175, "bottom": 135, "width": 70, "height": 30}, center=(140, 120),
                           words=[OCRWord(text="Save", rect={"left": 105, "top": 105, "right": 175, "bottom": 135, "width": 70, "height": 30}, center=(140, 120))])]
        )
        state = fuse_perception(mock_screen, mock_cursor, mock_window, uia_elems, ocr_res)
        assert len(state.targets) == 1
        tgt = state.targets[0]
        assert tgt.name == "Save"
        assert tgt.sources == ["uia", "ocr"]
        assert tgt.ocr_text == "Save"
        assert len(state.contradictions) == 0

    def test_case_b_overlapping_but_disagreeing_text(self, mock_screen, mock_cursor, mock_window) -> None:
        """Case B: UIA 'Save' + OCR 'Cancel' + overlapping -> NOT merged, contradiction logged, UIA sources=['uia'], OCR sources=['ocr']."""
        uia_elems = [
            {"name": "Save", "control_type": "Button", "rect": {"left": 100, "top": 100, "right": 180, "bottom": 140, "width": 80, "height": 40}, "center": (140, 120)}
        ]
        ocr_res = OCRResult(
            status=OCR_SUCCESS_TEXT_FOUND, text="Cancel", word_count=1, line_count=1,
            lines=[OCRLine(text="Cancel", rect={"left": 105, "top": 105, "right": 175, "bottom": 135, "width": 70, "height": 30}, center=(140, 120),
                           words=[OCRWord(text="Cancel", rect={"left": 105, "top": 105, "right": 175, "bottom": 135, "width": 70, "height": 30}, center=(140, 120))])]
        )
        state = fuse_perception(mock_screen, mock_cursor, mock_window, uia_elems, ocr_res)
        # Must produce 2 targets: one UIA 'Save', one OCR 'Cancel'
        assert len(state.targets) == 2
        uia_tgt = next(t for t in state.targets if t.name == "Save")
        ocr_tgt = next(t for t in state.targets if t.name == "Cancel")
        assert uia_tgt.sources == ["uia"]
        assert ocr_tgt.sources == ["ocr"]
        # Contradiction MUST be logged!
        assert len(state.contradictions) == 1
        assert "UIA/OCR text disagreement" in state.contradictions[0]
        assert "Save" in state.contradictions[0] and "Cancel" in state.contradictions[0]

    def test_case_c_matching_text_but_non_overlapping(self, mock_screen, mock_cursor, mock_window) -> None:
        """Case C: UIA 'Save' + OCR 'Save' + disjoint coordinates -> NOT merged, 2 separate targets."""
        uia_elems = [
            {"name": "Save", "control_type": "Button", "rect": {"left": 100, "top": 100, "right": 180, "bottom": 140, "width": 80, "height": 40}, "center": (140, 120)}
        ]
        ocr_res = OCRResult(
            status=OCR_SUCCESS_TEXT_FOUND, text="Save", word_count=1, line_count=1,
            lines=[OCRLine(text="Save", rect={"left": 500, "top": 500, "right": 580, "bottom": 540, "width": 80, "height": 40}, center=(540, 520),
                           words=[OCRWord(text="Save", rect={"left": 500, "top": 500, "right": 580, "bottom": 540, "width": 80, "height": 40}, center=(540, 520))])]
        )
        state = fuse_perception(mock_screen, mock_cursor, mock_window, uia_elems, ocr_res)
        assert len(state.targets) == 2
        t1, t2 = state.targets[0], state.targets[1]
        assert t1.sources == ["uia"]
        assert t2.sources == ["ocr"]
        assert len(state.contradictions) == 0

    def test_case_d_asymmetric_token_containment(self, mock_screen, mock_cursor, mock_window) -> None:
        """Case D: UIA 'Save' + OCR 'Save As' + overlapping -> corroborated with asymmetric token containment."""
        uia_elems = [
            {"name": "Save", "control_type": "Button", "rect": {"left": 100, "top": 100, "right": 220, "bottom": 140, "width": 120, "height": 40}, "center": (160, 120)}
        ]
        ocr_res = OCRResult(
            status=OCR_SUCCESS_TEXT_FOUND, text="Save As", word_count=2, line_count=1,
            lines=[OCRLine(text="Save As", rect={"left": 105, "top": 105, "right": 215, "bottom": 135, "width": 110, "height": 30}, center=(160, 120),
                           words=[OCRWord(text="Save", rect={"left": 105, "top": 105, "right": 150, "bottom": 135, "width": 45, "height": 30}, center=(127, 120)),
                                  OCRWord(text="As", rect={"left": 160, "top": 105, "right": 215, "bottom": 135, "width": 55, "height": 30}, center=(187, 120))])]
        )
        state = fuse_perception(mock_screen, mock_cursor, mock_window, uia_elems, ocr_res)
        assert len(state.targets) == 1
        assert state.targets[0].sources == ["uia", "ocr"]
        assert state.targets[0].ocr_text == "Save As"
        assert len(state.contradictions) == 0

    def test_case_e_multi_word_label_overlapping_ocr_words(self, mock_screen, mock_cursor, mock_window) -> None:
        """Case E: Multi-word label UIA 'Save As' + OCR words ['Save', 'As'] overlapping -> corroborated!"""
        uia_elems = [
            {"name": "Save As", "control_type": "Button", "rect": {"left": 100, "top": 100, "right": 220, "bottom": 140, "width": 120, "height": 40}, "center": (160, 120)}
        ]
        ocr_res = OCRResult(
            status=OCR_SUCCESS_TEXT_FOUND, text="Save As", word_count=2, line_count=1,
            lines=[OCRLine(text="Save As", rect={"left": 105, "top": 105, "right": 215, "bottom": 135, "width": 110, "height": 30}, center=(160, 120),
                           words=[OCRWord(text="Save", rect={"left": 105, "top": 105, "right": 150, "bottom": 135, "width": 45, "height": 30}, center=(127, 120)),
                                  OCRWord(text="As", rect={"left": 160, "top": 105, "right": 215, "bottom": 135, "width": 55, "height": 30}, center=(187, 120))])]
        )
        state = fuse_perception(mock_screen, mock_cursor, mock_window, uia_elems, ocr_res)
        assert len(state.targets) == 1
        assert state.targets[0].name == "Save As"
        assert state.targets[0].sources == ["uia", "ocr"]
        assert state.targets[0].ocr_text == "Save As"

    def test_case_f_auto_ocr_skips_when_uia_has_target_runs_when_missing(self) -> None:
        """Case F: Auto OCR mode skips OCR when target found in UIA; runs OCR when target missing."""
        from unittest.mock import MagicMock
        from agent.tools.perception import PerceptionEngine

        mock_uia = MagicMock()
        mock_ocr = MagicMock()
        engine = PerceptionEngine(uia_client=mock_uia, ocr_engine=mock_ocr)

        # 1. Target present in UIA -> OCR should NOT be called
        mock_uia.get_active_window_elements.return_value = {
            "elements": [
                {
                    "name": "SaveButton",
                    "control_type": "Button",
                    "rect": {"left": 10, "top": 10, "right": 50, "bottom": 30, "width": 40, "height": 20},
                    "center": (30, 20),
                }
            ]
        }
        engine.observe(
            screen_size=(1920, 1080),
            cursor_pos=(0, 0),
            active_window_info={"hwnd": 123},
            ocr_mode="auto",
            target_query="SaveButton",
        )
        assert mock_ocr.recognize_region.call_count == 0
        assert mock_ocr.recognize_screen.call_count == 0

        # 2. Target missing from UIA -> OCR MUST be called
        mock_uia.get_active_window_elements.return_value = {"elements": []}
        mock_ocr.recognize_region.return_value = OCRResult(
            status=OCR_SUCCESS_TEXT_FOUND, text="CanvasTarget", word_count=1, line_count=1, lines=[]
        )
        engine.observe(
            screen_size=(1920, 1080),
            cursor_pos=(0, 0),
            active_window_info={"hwnd": 123, "rect": {"left": 0, "top": 0, "width": 500, "height": 500}},
            ocr_mode="auto",
            target_query="CanvasTarget",
        )
        assert mock_ocr.recognize_region.call_count == 1


class TestTargetDisambiguation:
    """Test deterministic priority resolution and ambiguity safety gating."""

    @pytest.fixture
    def target_pool(self) -> List[UnifiedTarget]:
        return [
            # 1. Corroborated target "Save"
            UnifiedTarget(
                id="tgt_save",
                name="Save",
                control_type="Button",
                rect={"left": 100, "top": 100, "right": 180, "bottom": 140, "width": 80, "height": 40},
                center=(140, 120),
                hwnd=101,
                sources=["uia", "ocr"],
                ocr_text="Save",
            ),
            # 2. UIA-only target "Save As"
            UnifiedTarget(
                id="tgt_save_as",
                name="Save As",
                control_type="Button",
                rect={"left": 100, "top": 160, "right": 180, "bottom": 200, "width": 80, "height": 40},
                center=(140, 180),
                hwnd=101,
                sources=["uia"],
            ),
            # 3. Two ambiguous identical "Delete" buttons
            UnifiedTarget(
                id="tgt_del_1",
                name="Delete",
                control_type="Button",
                rect={"left": 200, "top": 100, "right": 280, "bottom": 140, "width": 80, "height": 40},
                center=(240, 120),
                hwnd=101,
                sources=["uia"],
            ),
            UnifiedTarget(
                id="tgt_del_2",
                name="Delete",
                control_type="Button",
                rect={"left": 200, "top": 160, "right": 280, "bottom": 200, "width": 80, "height": 40},
                center=(240, 180),
                hwnd=101,
                sources=["uia"],
            ),
            # 4. OCR-only target "Help"
            UnifiedTarget(
                id="tgt_help",
                name="Help",
                control_type="Text",
                rect={"left": 300, "top": 100, "right": 360, "bottom": 130, "width": 60, "height": 30},
                center=(330, 115),
                hwnd=101,
                sources=["ocr"],
                ocr_text="Help",
            ),
        ]

    def test_corroborated_exact_match_priority(self, target_pool) -> None:
        """Corroborated exact match has highest priority."""
        res = resolve_target(target_pool, "Save")
        assert res.status == "RESOLVED"
        assert res.target is not None
        assert res.target.id == "tgt_save"
        assert "Exact UIA+OCR corroborated match" in res.reason

    def test_ambiguity_detection_multiple_candidates(self, target_pool) -> None:
        """Multiple candidates tying for highest priority tier triggers AMBIGUOUS status."""
        res = resolve_target(target_pool, "Delete")
        assert res.status == "AMBIGUOUS"
        assert res.target is None
        assert len(res.candidates) == 2
        assert "Ambiguous target: found 2 candidates" in res.reason

    def test_ocr_only_target_resolution(self, target_pool) -> None:
        """OCR-only target resolves cleanly when no UIA target exists."""
        res = resolve_target(target_pool, "Help")
        assert res.status == "RESOLVED"
        assert res.target is not None
        assert res.target.id == "tgt_help"
        assert res.target.sources == ["ocr"]

    def test_target_not_found(self, target_pool) -> None:
        """Non-existent query returns NOT_FOUND."""
        res = resolve_target(target_pool, "NonExistentButtonXYZ")
        assert res.status == "NOT_FOUND"
        assert res.target is None


class TestComputerToolObserveSemantic:
    """Test ComputerTool 'observe_semantic' action and policy classification."""

    def test_security_policy_classification(self) -> None:
        """observe_semantic must be deterministically classified as SAFE."""
        policy = SecurityPolicy()
        eval_res = policy.evaluate_action(
            tool_name="computer",
            arguments={"action": "observe_semantic", "ocr_mode": "off"},
            known_tool_names={"computer"},
        )
        assert eval_res.level == PermissionLevel.SAFE
        assert eval_res.is_blocked is False

    def test_observe_semantic_execution_off(self) -> None:
        """Execute observe_semantic with ocr_mode='off' on real host."""
        comp = ComputerTool()
        res = comp.execute({"action": "observe_semantic", "ocr_mode": "off", "max_elements": 20})
        assert res.success is True
        out = res.output
        assert "targets" in out
        assert "screen" in out
        assert "cursor" in out
        assert "active_window" in out
        assert out["ocr_performed"] is False


class TestVerifierPerceptionAssertions:
    """Test verifier semantic assertions on fused perception state."""

    @pytest.fixture
    def mock_perception_output(self) -> Dict[str, Any]:
        return {
            "timestamp": 1234567.89,
            "screen": {"width": 1920, "height": 1080},
            "cursor": (500, 500),
            "active_window": {"title": "Notepad - Test", "hwnd": 5555},
            "targets": [
                {
                    "id": "tgt_001_file",
                    "name": "File",
                    "control_type": "MenuItem",
                    "rect": {"left": 10, "top": 10, "right": 50, "bottom": 30, "width": 40, "height": 20},
                    "center": (30, 20),
                    "hwnd": 5555,
                    "sources": ["uia", "ocr"],
                    "ocr_text": "File",
                },
                {
                    "id": "tgt_002_canvas",
                    "name": "Special Canvas Label",
                    "control_type": "Text",
                    "rect": {"left": 100, "top": 100, "right": 300, "bottom": 130, "width": 200, "height": 30},
                    "center": (200, 115),
                    "hwnd": 5555,
                    "sources": ["ocr"],
                    "ocr_text": "Special Canvas Label",
                },
            ],
            "contradictions": [],
        }

    def test_expected_target_present_pass(self, mock_perception_output) -> None:
        tool_res = ToolResult(success=True, output=mock_perception_output)
        record = default_verifier.verify(
            "computer",
            {"action": "observe_semantic", "expected_target_present": "File"},
            tool_res,
        )
        assert record.status == VerificationStatus.VERIFIED

    def test_expected_target_present_fail(self, mock_perception_output) -> None:
        tool_res = ToolResult(success=True, output=mock_perception_output)
        record = default_verifier.verify(
            "computer",
            {"action": "observe_semantic", "expected_target_present": "MissingTargetBtn"},
            tool_res,
        )
        assert record.status == VerificationStatus.FAILED
        assert "was not found in perception targets" in record.verification

    def test_expected_target_source_verification(self, mock_perception_output) -> None:
        tool_res = ToolResult(success=True, output=mock_perception_output)
        # Passing: "Special Canvas Label" has source ["ocr"]
        rec_pass = default_verifier.verify(
            "computer",
            {
                "action": "observe_semantic",
                "expected_target_present": "Special Canvas Label",
                "expected_target_source": "ocr",
            },
            tool_res,
        )
        assert rec_pass.status == VerificationStatus.VERIFIED

        # Failing: "Special Canvas Label" does not have source "uia"
        rec_fail = default_verifier.verify(
            "computer",
            {
                "action": "observe_semantic",
                "expected_target_present": "Special Canvas Label",
                "expected_target_source": "uia",
            },
            tool_res,
        )
        assert rec_fail.status == VerificationStatus.FAILED
        assert "which does not include 'uia'" in rec_fail.verification

    def test_expected_target_absent(self, mock_perception_output) -> None:
        tool_res = ToolResult(success=True, output=mock_perception_output)
        # Absent target passes
        rec_pass = default_verifier.verify(
            "computer",
            {"action": "observe_semantic", "expected_target_absent": "NonExistentMenu"},
            tool_res,
        )
        assert rec_pass.status == VerificationStatus.VERIFIED

        # Present target fails when asserted absent
        rec_fail = default_verifier.verify(
            "computer",
            {"action": "observe_semantic", "expected_target_absent": "File"},
            tool_res,
        )
        assert rec_fail.status == VerificationStatus.FAILED

    def test_expected_min_targets(self, mock_perception_output) -> None:
        tool_res = ToolResult(success=True, output=mock_perception_output)
        rec_pass = default_verifier.verify(
            "computer",
            {"action": "observe_semantic", "expected_min_targets": 2},
            tool_res,
        )
        assert rec_pass.status == VerificationStatus.VERIFIED

        rec_fail = default_verifier.verify(
            "computer",
            {"action": "observe_semantic", "expected_min_targets": 5},
            tool_res,
        )
        assert rec_fail.status == VerificationStatus.FAILED
        assert "Expected at least 5 targets" in rec_fail.verification

    def test_disagreement_detection(self, mock_perception_output) -> None:
        # State with no contradiction passes expected_perception_disagreement=False
        tool_res = ToolResult(success=True, output=mock_perception_output)
        rec = default_verifier.verify(
            "computer",
            {"action": "observe_semantic", "expected_perception_disagreement": False},
            tool_res,
        )
        assert rec.status == VerificationStatus.VERIFIED

        # Contradiction present
        conflicted_output = dict(mock_perception_output)
        conflicted_output["contradictions"] = ["UIA reports element missing, OCR reports element present."]
        tool_res_conflict = ToolResult(success=True, output=conflicted_output)

        rec_detect = default_verifier.verify(
            "computer",
            {"action": "observe_semantic", "expected_perception_disagreement": True},
            tool_res_conflict,
        )
        assert rec_detect.status == VerificationStatus.VERIFIED

        rec_fail = default_verifier.verify(
            "computer",
            {"action": "observe_semantic", "expected_perception_disagreement": False},
            tool_res_conflict,
        )
        assert rec_fail.status == VerificationStatus.FAILED


class TestAgentPerceptionIntegration:
    """Test Agent loop integration with fused perception and deterministic safety gates."""

    def test_agent_ambiguity_gating_aborts_action(self) -> None:
        """When target resolution encounters ambiguous candidates, action is aborted with failure."""
        from unittest.mock import MagicMock
        from agent.core.agent import Agent
        from agent.core.planner import Plan, PlanStep
        from agent.core.state import StepResult, TaskStateEnum

        from unittest.mock import MagicMock
        from agent.core.agent import Agent
        from agent.core.planner import Plan, PlanStep
        from agent.core.state import StepResult, TaskStateEnum
        from agent.tools.registry import ToolRegistry

        prior_targets = [
            {
                "id": "tgt_ambig_1",
                "name": "CloneBtn",
                "control_type": "Button",
                "rect": {"left": 100, "top": 100, "right": 180, "bottom": 140, "width": 80, "height": 40},
                "center": (140, 120),
                "hwnd": 9999,
                "sources": ["uia"],
            },
            {
                "id": "tgt_ambig_2",
                "name": "CloneBtn",
                "control_type": "Button",
                "rect": {"left": 100, "top": 200, "right": 180, "bottom": 240, "width": 80, "height": 40},
                "center": (140, 220),
                "hwnd": 9999,
                "sources": ["uia"],
            },
        ]

        mock_tool = MagicMock()
        mock_tool.name = "computer"
        mock_reg = MagicMock()
        mock_reg.execute.return_value = ToolResult(
            success=True,
            output={"targets": prior_targets, "active_window": {"hwnd": 9999}},
        )
        mock_reg.list_tools.return_value = [mock_tool]

        plan = Plan(
            goal="Test Ambiguous Goal",
            steps=[
                PlanStep(
                    step_id="step_1",
                    objective="Observe semantic targets",
                    tool_required="computer",
                    arguments={"action": "observe_semantic"},
                ),
                PlanStep(
                    step_id="step_2",
                    objective="Click ambiguous button",
                    tool_required="computer",
                    arguments={"action": "mouse_click", "target_element": "CloneBtn"},
                ),
            ]
        )

        planner = MagicMock()
        planner.create_plan.return_value = plan
        agent = Agent(planner=planner, tool_registry=mock_reg)

        state = agent.run("Perform ambiguous interaction")
        assert state.status == TaskStateEnum.FAILED
        # Verify action was aborted due to ambiguity
        ambig_actions = [a for a in state.actions if "Ambiguous target" in (a.error or "")]
        assert len(ambig_actions) == 1
        assert "found 2 candidates" in (ambig_actions[0].error or "")

