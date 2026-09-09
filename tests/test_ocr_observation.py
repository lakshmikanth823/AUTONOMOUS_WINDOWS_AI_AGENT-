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

    def test_coordinate_offset_translation(self, sample_text_image: Path) -> None:
        ocr = WindowsNativeOCR()
        offset_x, offset_y = 500, 300
        res = ocr.recognize_image(sample_text_image, offset=(offset_x, offset_y))
        assert res.status == OCR_SUCCESS_TEXT_FOUND
        assert len(res.lines) > 0

        first_word = res.lines[0].words[0]
        # Bounding rect must be translated by offset
        assert first_word.rect["left"] >= offset_x
        assert first_word.rect["top"] >= offset_y
        assert first_word.center[0] >= offset_x
        assert first_word.center[1] >= offset_y

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
