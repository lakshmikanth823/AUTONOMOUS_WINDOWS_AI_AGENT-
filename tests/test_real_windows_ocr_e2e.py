"""Real-world E2E demonstration of Native Windows OCR observation, window rendering, and semantic verification."""

import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.core.verifier import VerificationStatus, default_verifier
from agent.tools.computer import ComputerTool


def main() -> None:
    comp = ComputerTool()
    expected_content = "Autonomous OCR Verified 2026"

    print("=== 1. Launch Real Notepad Window ===")
    proc = subprocess.Popen(["notepad.exe"])
    time.sleep(2.0)
    notepad_hwnd = None

    try:
        # Focus window to bring it to foreground
        focus_res = comp.execute({"action": "window_focus", "text": "Notepad"})
        print("window_focus success:", focus_res.success)
        assert focus_res.success, f"window_focus failed: {focus_res.error}"
        notepad_hwnd = focus_res.output.get("hwnd")
        print("Notepad HWND:", notepad_hwnd)
        assert notepad_hwnd is not None, "Expected valid window HWND"

        # Set text on the active editor surface
        print("\n=== 2. Set Text via Element ===")
        set_res = comp.execute({
            "action": "set_element_text",
            "text": expected_content,
            "target_element": "Text editor",
            "hwnd": notepad_hwnd,
        })
        print("set_element_text success:", set_res.success)
        assert set_res.success is True, f"set_element_text failed: {set_res.error}"

        time.sleep(0.5)

        # ---------------------------------------------------------------------
        # 3. Retrieve Window Bounds & Perform Region OCR
        # ---------------------------------------------------------------------
        print("\n=== 3. Retrieve Window Bounds & Perform Window-Bound Region OCR ===")
        win_info = comp._get_active_window_info()
        rect = win_info.get("rect", {})
        rx = max(0, int(rect.get("left", 0)))
        ry = max(0, int(rect.get("top", 0)))
        rw = int(rect.get("width", 600))
        rh = int(rect.get("height", 300))
        print(f"Active window rect: ({rx}, {ry}, {rw}, {rh})")

        ocr_res = comp.execute({
            "action": "ocr_region",
            "x": rx,
            "y": ry,
            "width": rw,
            "height": rh,
            "hwnd": notepad_hwnd,
        })
        print("ocr_region success:", ocr_res.success)
        assert ocr_res.success is True, f"ocr_region failed: {ocr_res.error}"

        out = ocr_res.output
        clean_text = out.get("text", "").encode("ascii", "replace").decode("ascii")
        print("OCR Recognized Text:", repr(clean_text))
        print("OCR Status:", out.get("status"))
        print("Word count:", out.get("word_count"))
        print("Line count:", out.get("line_count"))

        assert out.get("status") == "OCR_SUCCESS_TEXT_FOUND", "Expected text found"
        assert out.get("word_count", 0) >= 3, "Expected at least 3 words recognized"
        assert "OCR" in clean_text or "Verified" in clean_text or "2026" in clean_text

        # Validate that recognized words have absolute coordinates matching the region
        words = []
        for line in out.get("lines", []):
            for w in line.get("words", []):
                words.append(w)
                assert w["rect"]["left"] >= rx - 5, f"Word coord {w['rect']['left']} outside window left {rx}"
                assert w["rect"]["top"] >= ry - 5, f"Word coord {w['rect']['top']} outside window top {ry}"

        print(f"Verified {len(words)} words all have valid absolute screen coordinates.")

        # ---------------------------------------------------------------------
        # 4. Semantic Verification: Positive Assertions
        # ---------------------------------------------------------------------
        print("\n=== 4. Semantic Verification: Positive Assertions ===")
        pos_verif_args = {
            "action": "ocr_region",
            "expected_ocr_text_present": "OCR Verified",
            "expected_ocr_word": "Verified",
            "expected_ocr_text_absent": "PhonyStringThatDoesNotExist_404",
        }
        pos_record = default_verifier.verify("computer", pos_verif_args, ocr_res)
        print("Positive verification status:", pos_record.status)
        print("Positive verification details:", pos_record.verification)
        assert pos_record.status == VerificationStatus.VERIFIED

        # ---------------------------------------------------------------------
        # 5. Semantic Verification: Negative Assertions (Must Fail)
        # ---------------------------------------------------------------------
        print("\n=== 5. Semantic Verification: Negative Assertions ===")
        # Negative A: asserting absent text that IS present must fail
        neg_a = default_verifier.verify(
            "computer",
            {"action": "ocr_region", "expected_ocr_text_absent": "Verified"},
            ocr_res,
        )
        print("Negative assertion A status (expected FAILED):", neg_a.status)
        assert neg_a.status == VerificationStatus.FAILED

        # Negative B: asserting imaginary text is present must fail
        neg_b = default_verifier.verify(
            "computer",
            {"action": "ocr_region", "expected_ocr_text_present": "ImaginaryTextNotOnScreen_12345"},
            ocr_res,
        )
        print("Negative assertion B status (expected FAILED):", neg_b.status)
        assert neg_b.status == VerificationStatus.FAILED

        # Negative C: asserting wrong exact text must fail
        neg_c = default_verifier.verify(
            "computer",
            {"action": "ocr_region", "expected_ocr_exact_text": "Completely Wrong Exact String"},
            ocr_res,
        )
        print("Negative assertion C status (expected FAILED):", neg_c.status)
        assert neg_c.status == VerificationStatus.FAILED

        print("All negative assertions correctly rejected invalid expectations.")

    finally:
        # ---------------------------------------------------------------------
        # 6. Cleanup & Verification
        # ---------------------------------------------------------------------
        print("\n=== 6. Close Notepad & Cleanup Verification ===")
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(1.0)
        print("Notepad process terminated successfully.")

    # Verify window is no longer active
    post_obs = comp.execute({"action": "window_list"})
    post_windows = [str(w).lower() for w in post_obs.output.get("windows", [])]
    notepad_remaining = any("untitled - notepad" in w for w in post_windows)
    print("Notepad remaining in window list:", notepad_remaining)
    assert not notepad_remaining, "Expected Notepad window to be cleaned up after termination"

    print("\nALL REAL-WINDOWS OCR E2E CHECKS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
