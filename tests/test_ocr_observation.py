"""Unit and integration tests for Native Windows OCR observation, verifier assertions, and security."""

import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pytest

from agent.config.permissions import PermissionLevel
from agent.core.verifier import VerificationStatus, default_verifier
from agent.security.policy import default_security_policy
from agent.tools.base import ToolResult
from agent.tools.computer import ComputerTool
from agent.tools.ocr import (
    OCR_FAILED,
    OCR_SUCCESS_NO_TEXT,
    OCR_SUCCESS_TEXT_FOUND,
    WindowsNativeOCR,
)


@pytest.fixture
def sample_text_image(tmp_path: Path) -> Path:
    """Create a temporary high-contrast image with known text."""
    img_path = tmp_path / "sample_text.png"
    img = Image.new("RGB", (600, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    draw.text((30, 40), "Windows Native OCR 2026", fill=(0, 0, 0), font=font)
    img.save(img_path)
    return img_path


@pytest.fixture
def blank_image(tmp_path: Path) -> Path:
    """Create a temporary blank white image."""
    img_path = tmp_path / "blank.png"
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    img.save(img_path)
    return img_path


class TestWindowsNativeOCR:
    """Direct tests for WindowsNativeOCR engine."""

    def test_recognize_image_positive(self, sample_text_image: Path) -> None:
        ocr = WindowsNativeOCR()
        res = ocr.recognize_image(sample_text_image)
        assert res.status == OCR_SUCCESS_TEXT_FOUND
        assert "Windows" in res.text
        assert "OCR" in res.text
        assert res.word_count > 0
        assert res.line_count > 0

        # Check word structure
        first_word = res.lines[0].words[0]
        assert first_word.text.lower() in ("windows", "native", "ocr", "2026")
        assert first_word.rect["width"] > 0
        assert first_word.rect["height"] > 0
        assert len(first_word.center) == 2

    def test_coordinate_offset_translation_formula_proof(self, sample_text_image: Path) -> None:
        """Prove absolute_x = region_x + local_x and absolute_y = region_y + local_y mathematically."""
        ocr = WindowsNativeOCR()
        base_res = ocr.recognize_image(sample_text_image, offset=(0, 0))
        assert base_res.status == OCR_SUCCESS_TEXT_FOUND
        assert len(base_res.lines) > 0

        reg_x, reg_y = 345, 678
        offset_res = ocr.recognize_image(sample_text_image, offset=(reg_x, reg_y))
        assert offset_res.status == OCR_SUCCESS_TEXT_FOUND

        base_words = base_res.lines[0].words
        offset_words = offset_res.lines[0].words
        assert len(base_words) == len(offset_words)

        for bw, ow in zip(base_words, offset_words):
            assert ow.text == bw.text
            # Mathematical proof: absolute = region + local
            assert ow.rect["left"] == reg_x + bw.rect["left"]
            assert ow.rect["top"] == reg_y + bw.rect["top"]
            assert ow.rect["right"] == reg_x + bw.rect["right"]
            assert ow.rect["bottom"] == reg_y + bw.rect["bottom"]
            assert ow.center[0] == reg_x + bw.center[0]
            assert ow.center[1] == reg_y + bw.center[1]

    def test_corrupted_image_failure_mode(self, tmp_path: Path) -> None:
        """Verify that corrupted image input returns OCR_FAILED, never OCR_SUCCESS_NO_TEXT."""
        corrupt_file = tmp_path / "corrupted.png"
        corrupt_file.write_bytes(b"\x89PNG\r\n\x1a\nCORRUPTED_GARBAGE_PAYLOAD_NOT_AN_IMAGE")

        ocr = WindowsNativeOCR()
        res = ocr.recognize_image(corrupt_file)
        assert res.status == OCR_FAILED
        assert res.status != OCR_SUCCESS_NO_TEXT
        assert res.error is not None

    def test_negative_region_coordinates(self) -> None:
        """Reject negative region coordinates."""
        ocr = WindowsNativeOCR()
        res = ocr.recognize_region(-50, 20, 100, 100)
        assert res.status == OCR_FAILED
        assert "cannot be negative" in (res.error or "")

    def test_region_near_screen_edge(self) -> None:
        """Capture region near screen boundary without exception."""
        ocr = WindowsNativeOCR()
        res = ocr.recognize_region(1800, 1000, 100, 50)
        assert res.status in (OCR_SUCCESS_TEXT_FOUND, OCR_SUCCESS_NO_TEXT)
        assert res.region == (1800, 1000, 1900, 1050)

    def test_blank_image_no_text(self, blank_image: Path) -> None:
        ocr = WindowsNativeOCR()
        res = ocr.recognize_image(blank_image)
        assert res.status == OCR_SUCCESS_NO_TEXT
        assert res.text == ""
        assert len(res.lines) == 0
        assert res.word_count == 0

    def test_missing_image_file(self, tmp_path: Path) -> None:
        ocr = WindowsNativeOCR()
        missing = tmp_path / "non_existent_file_987654.png"
        res = ocr.recognize_image(missing)
        assert res.status == OCR_FAILED
        assert res.error is not None
        assert "does not exist" in res.error

    def test_invalid_region_dimensions(self) -> None:
        ocr = WindowsNativeOCR()
        res = ocr.recognize_region(10, 10, -50, 100)
        assert res.status == OCR_FAILED
        assert "Invalid region dimensions" in (res.error or "")


class TestComputerToolOCRIntegration:
    """Tests for ComputerTool ocr_screen and ocr_region actions."""

    def test_ocr_region_execution(self) -> None:
        comp = ComputerTool()
        res = comp.execute({
            "action": "ocr_region",
            "x": 0,
            "y": 0,
            "width": 150,
            "height": 80,
        })
        assert res.success is True
        assert isinstance(res.output, dict)
        assert "status" in res.output
        assert "text" in res.output
        assert "lines" in res.output
        assert tuple(res.output.get("region")) == (0, 0, 150, 80)

    def test_ocr_region_missing_args(self) -> None:
        comp = ComputerTool()
        res = comp.execute({"action": "ocr_region", "x": 0})
        assert res.success is False
        assert "Parameters 'x' and 'y' are required" in (res.error or "")

        res2 = comp.execute({"action": "ocr_region", "x": 0, "y": 0})
        assert res2.success is False
        assert "Parameters 'width' and 'height' are required" in (res2.error or "")

    def test_ocr_screen_execution(self) -> None:
        comp = ComputerTool()
        res = comp.execute({"action": "ocr_screen"})
        assert res.success is True
        assert isinstance(res.output, dict)
        assert "status" in res.output
        assert "lines" in res.output


class TestVerifierOCRAssertions:
    """Tests for semantic OCR verifier assertions."""

    def test_expected_ocr_text_present_pass(self, sample_text_image: Path) -> None:
        ocr = WindowsNativeOCR()
        ocr_res = ocr.recognize_image(sample_text_image)
        tool_result = ToolResult(success=True, output=ocr_res.model_dump())

        record = default_verifier.verify(
            "computer",
            {"action": "ocr_region", "expected_ocr_text_present": "Windows Native"},
            tool_result,
        )
        assert record.status == VerificationStatus.VERIFIED
        assert "All semantic OCR assertions satisfied" in record.verification

    def test_expected_ocr_text_present_fail(self, sample_text_image: Path) -> None:
        ocr = WindowsNativeOCR()
        ocr_res = ocr.recognize_image(sample_text_image)
        tool_result = ToolResult(success=True, output=ocr_res.model_dump())

        record = default_verifier.verify(
            "computer",
            {"action": "ocr_region", "expected_ocr_text_present": "ImaginaryNonExistentString_777"},
            tool_result,
        )
        assert record.status == VerificationStatus.FAILED
        assert "was not found in recognized text" in record.verification

    def test_expected_ocr_exact_text_pass_and_fail(self, sample_text_image: Path) -> None:
        ocr = WindowsNativeOCR()
        ocr_res = ocr.recognize_image(sample_text_image)
        actual_text = ocr_res.text.strip()
        tool_result = ToolResult(success=True, output=ocr_res.model_dump())

        # Exact match pass
        pass_record = default_verifier.verify(
            "computer",
            {"action": "ocr_region", "expected_ocr_exact_text": actual_text},
            tool_result,
        )
        assert pass_record.status == VerificationStatus.VERIFIED

        # Exact match fail
        fail_record = default_verifier.verify(
            "computer",
            {"action": "ocr_region", "expected_ocr_exact_text": "Different Exact Text"},
            tool_result,
        )
        assert fail_record.status == VerificationStatus.FAILED
        assert "Expected exact OCR text" in fail_record.verification

    def test_expected_ocr_text_absent(self, sample_text_image: Path) -> None:
        ocr = WindowsNativeOCR()
        ocr_res = ocr.recognize_image(sample_text_image)
        tool_result = ToolResult(success=True, output=ocr_res.model_dump())

        # Absence verified pass
        pass_record = default_verifier.verify(
            "computer",
            {"action": "ocr_region", "expected_ocr_text_absent": "ConfidentialSecretXYZ"},
            tool_result,
        )
        assert pass_record.status == VerificationStatus.VERIFIED

        # Absence fail when text is present
        fail_record = default_verifier.verify(
            "computer",
            {"action": "ocr_region", "expected_ocr_text_absent": "Windows"},
            tool_result,
        )
        assert fail_record.status == VerificationStatus.FAILED
        assert "to be absent, but it was found" in fail_record.verification

    def test_expected_ocr_word(self, sample_text_image: Path) -> None:
        ocr = WindowsNativeOCR()
        ocr_res = ocr.recognize_image(sample_text_image)
        tool_result = ToolResult(success=True, output=ocr_res.model_dump())

        pass_record = default_verifier.verify(
            "computer",
            {"action": "ocr_region", "expected_ocr_word": "OCR"},
            tool_result,
        )
        assert pass_record.status == VerificationStatus.VERIFIED

        fail_record = default_verifier.verify(
            "computer",
            {"action": "ocr_region", "expected_ocr_word": "NonExistentWord123"},
            tool_result,
        )
        assert fail_record.status == VerificationStatus.FAILED
        assert "was not found in recognized words" in fail_record.verification


class TestSecurityPolicyOCR:
    """Security classification tests for OCR actions."""

    def test_ocr_actions_classified_safe(self) -> None:
        eval_screen = default_security_policy.evaluate_action(
            "computer", {"action": "ocr_screen"}, known_tool_names={"computer"}
        )
        assert eval_screen.level == PermissionLevel.SAFE
        assert eval_screen.is_blocked is False

        eval_region = default_security_policy.evaluate_action(
            "computer",
            {"action": "ocr_region", "x": 10, "y": 10, "width": 100, "height": 50},
            known_tool_names={"computer"},
        )
        assert eval_region.level == PermissionLevel.SAFE
        assert eval_region.is_blocked is False


class TestOCRSafetyAndVerification:
    """Hardened tests for OCR target safety and semantic comparison integrity."""

    def test_stale_target_rejection_on_mouse_action(self) -> None:
        comp = ComputerTool()
        stale_hwnd = 0x7FFFFFFF
        res = comp.execute({
            "action": "mouse_click",
            "x": 200,
            "y": 200,
            "expected_hwnd": stale_hwnd,
        })
        assert res.success is False
        assert "Stale target safety violation" in (res.error or "")

    def test_valid_target_mouse_action_proceeds(self) -> None:
        comp = ComputerTool()
        active_hwnd = comp._get_active_window_info().get("hwnd")
        if active_hwnd:
            res = comp.execute({
                "action": "mouse_move",
                "x": 200,
                "y": 200,
                "expected_hwnd": active_hwnd,
            })
            assert res.success is True

    def test_semantic_verification_requires_actual_comparison(self) -> None:
        """API success alone must never verify if semantic expectation fails."""
        tool_res = ToolResult(
            success=True,
            output={
                "status": OCR_SUCCESS_TEXT_FOUND,
                "text": "Completely Different Unrelated Content",
                "lines": [],
            },
        )
        record = default_verifier.verify(
            "computer",
            {"action": "ocr_screen", "expected_ocr_text_present": "TargetNeedleText"},
            tool_res,
        )
        assert record.status == VerificationStatus.FAILED
        assert record.passed is False
        assert "was not found in recognized text" in record.verification

    def test_ocr_failed_status_never_verified(self) -> None:
        """OCR_FAILED status in output must fail verification."""
        tool_res = ToolResult(
            success=True,
            output={
                "status": OCR_FAILED,
                "error": "Simulated hardware error",
                "text": "",
                "lines": [],
            },
        )
        record = default_verifier.verify(
            "computer",
            {"action": "ocr_screen"},
            tool_res,
        )
        assert record.status == VerificationStatus.FAILED
        assert "OCR execution failed" in record.verification

    def test_non_uia_canvas_text_extraction(self) -> None:
        """Prove that visible text drawn on a custom canvas invisible to UIA is recognized by OCR."""
        import subprocess
        import sys
        import textwrap
        import time
        import ctypes
        from agent.tools.uia import UIAClient

        code = textwrap.dedent("""
            import tkinter as tk
            import sys
            import ctypes
            root = tk.Tk()
            root.title('Non-UIA Canvas Harness')
            root.geometry('450x200+180+180')
            canvas = tk.Canvas(root, width=450, height=200, bg='white')
            canvas.pack(fill='both', expand=True)
            canvas.create_text(225, 100, text='Secret Canvas Text 2026', font=('Arial', 20, 'bold'), fill='black')
            root.update_idletasks()
            root.update()
            hwnd = ctypes.windll.user32.FindWindowW(None, 'Non-UIA Canvas Harness')
            sys.stdout.write(f"{hwnd}\\n")
            sys.stdout.flush()
            root.mainloop()
        """)
        proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        line = proc.stdout.readline().strip()
        assert line, f"Expected HWND from child process, got empty string. Stderr: {proc.stderr.read()}"
        hwnd = int(line)
        time.sleep(0.3)

        try:
            assert hwnd != 0, f"Expected valid canvas window HWND, proc poll: {proc.poll()}"

            # 1. Prove UIA fails to expose the canvas text
            uia = UIAClient()
            uia_res = uia.get_active_window_elements(hwnd=hwnd)
            uia_texts = [el.get("name", "") for el in uia_res.get("elements", [])] + [
                el.get("value", "") for el in uia_res.get("elements", []) if el.get("value")
            ]
            assert not any("Secret Canvas Text 2026" in t for t in uia_texts), "Canvas text must NOT be exposed to UIA"

            # 2. Prove OCR recognizes the non-UIA text with usable coordinates
            comp = ComputerTool()
            ocr_res = comp.execute({
                "action": "ocr_region",
                "x": 180,
                "y": 180,
                "width": 450,
                "height": 200,
                "hwnd": hwnd,
            })
            assert ocr_res.success is True
            assert ocr_res.output.get("status") == OCR_SUCCESS_TEXT_FOUND
            rec_text = ocr_res.output.get("text", "")
            assert "Secret Canvas Text 2026" in rec_text or "Canvas Text" in rec_text
        finally:
            proc.terminate()


