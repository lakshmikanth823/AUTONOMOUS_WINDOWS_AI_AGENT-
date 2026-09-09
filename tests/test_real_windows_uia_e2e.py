"""Real-world E2E demonstration of Windows UI Automation (UIA) observation, Notepad lifecycle, and semantic verification."""

import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.core.verifier import default_verifier, VerificationStatus
from agent.tools.computer import ComputerTool
from agent.tools.uia import UIAClient


def main() -> None:
    comp = ComputerTool()

    print("=== 1. Launch Notepad Process ===")
    proc = subprocess.Popen(["notepad.exe"])
    time.sleep(2.0)
    notepad_hwnd = None

    try:
        # Focus Notepad to ensure it is in foreground
        focus_res = comp.execute({"action": "window_focus", "text": "Notepad"})
        print("window_focus success:", focus_res.success)
        assert focus_res.success, f"window_focus failed: {focus_res.error}"
        notepad_hwnd = focus_res.output.get("hwnd")
        print("Notepad HWND:", notepad_hwnd)
        assert notepad_hwnd is not None, "Expected valid Notepad HWND"

        # ---------------------------------------------------------------------
        # 2. UIA Observation with Depth & Element Limits
        # ---------------------------------------------------------------------
        print("\n=== 2. UIA Observation (max_depth=5, max_elements=50) ===")
        obs_res = comp.execute({
            "action": "ui_elements",
            "hwnd": notepad_hwnd,
            "max_depth": 5,
            "max_elements": 50,
        })
        print("ui_elements success:", obs_res.success)
        assert obs_res.success is True, f"ui_elements failed: {obs_res.error}"

        out = obs_res.output
        win = out.get("window", {})
        clean_title = win.get("title", "").encode("ascii", "replace").decode("ascii")
        print("Active window title:", clean_title)
        print("Reported max_depth:", out.get("max_depth"))
        print("Discovered element count:", out.get("element_count"))
        assert out.get("max_depth") == 5, "Expected max_depth=5"
        assert out.get("element_count", 0) > 0, "Expected at least 1 UI element discovered"

        elements = out.get("elements", [])
        # Verify BFS depth contract: all elements have depth <= max_depth
        for el in elements:
            assert "depth" in el, "Each element must have depth attribute"
            assert el["depth"] <= 5, f"Element depth {el['depth']} exceeded max_depth 5"

        print(f"Verified {len(elements)} elements all satisfy depth <= 5.")

        # ---------------------------------------------------------------------
        # 3. Locate Edit / Document Control
        # ---------------------------------------------------------------------
        print("\n=== 3. Locate Edit Control ===")
        edit_elem = None
        for el in elements:
            if el.get("control_type") in ("Document", "Edit") or "text editor" in el.get("name", "").lower():
                edit_elem = el
                break

        assert edit_elem is not None, "Expected to locate Text Editor / Edit control in Notepad"
        clean_name = edit_elem.get("name", "").encode("ascii", "replace").decode("ascii")
        print(f"Located target: [{edit_elem.get('control_type')}] '{clean_name}' at center {edit_elem.get('center')}")

        # ---------------------------------------------------------------------
        # 4. Input Text via Target Element
        # ---------------------------------------------------------------------
        print("\n=== 4. Set/Type Input Text ===")
        expected_text = "Autonomous UIA Verified 2026"
        set_res = comp.execute({
            "action": "set_element_text",
            "text": expected_text,
            "target_element": "Text editor",
            "hwnd": notepad_hwnd,
        })
        print("set_element_text success:", set_res.success)
        assert set_res.success is True, f"set_element_text failed: {set_res.error}"

        # ---------------------------------------------------------------------
        # 5. Semantic State Verification & Text Readback
        # ---------------------------------------------------------------------
        print("\n=== 5. Semantic State Verification & Text Readback ===")
        # Re-observe state via fresh UIA observation
        reobs_res = comp.execute({
            "action": "ui_elements",
            "hwnd": notepad_hwnd,
            "max_depth": 5,
            "max_elements": 50,
        })
        assert reobs_res.success is True

        # Assert active window is Notepad, Close button is present, imaginary dialog is absent,
        # AND exact element text readback matches expected_text
        verif_args = {
            "action": "ui_elements",
            "expected_window_active": "Notepad",
            "expected_element_present": "Close",
            "expected_element_absent": "NonExistentDialog_87654321",
            "expected_element_text": {
                "element": "Text editor",
                "text": expected_text,
                "exact": True,
            },
        }
        verif_record = default_verifier.verify("computer", verif_args, reobs_res)
        print("Positive verification status:", verif_record.status)
        assert verif_record.status == VerificationStatus.VERIFIED, f"Verification failed: {verif_record.verification}"

        # Direct readback via read_element_text
        read_res = comp.execute({
            "action": "read_element_text",
            "target_element": "Text editor",
            "hwnd": notepad_hwnd,
        })
        print("Direct read_element_text success:", read_res.success, "text:", repr(read_res.output.get("text")))
        assert read_res.success is True
        assert read_res.output.get("text") == expected_text, f"Text mismatch: expected {expected_text}, got {read_res.output.get('text')}"

        # Negative assertion A: asserting non-existent element present must fail
        neg_record = default_verifier.verify(
            "computer",
            {"action": "ui_elements", "expected_element_present": "ImaginaryModalControlXYZ"},
            reobs_res,
        )
        print("Negative assertion status (expected FAILED):", neg_record.status)
        assert neg_record.status == VerificationStatus.FAILED

        # Negative assertion B: asserting wrong text in element must fail
        neg_text_record = default_verifier.verify(
            "computer",
            {
                "action": "ui_elements",
                "expected_element_text": {
                    "element": "Text editor",
                    "text": "Completely Incorrect Corrupted Content 999",
                    "exact": True,
                },
            },
            reobs_res,
        )
        print("Wrong text assertion status (expected FAILED):", neg_text_record.status)
        assert neg_text_record.status == VerificationStatus.FAILED

    finally:
        # ---------------------------------------------------------------------
        # 6. Close & Cleanup Verification
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

    # ---------------------------------------------------------------------
    # 7. Dynamic Target & Stale HWND Safety Enforcement
    # ---------------------------------------------------------------------
    print("\n=== 7. Stale Target Safety Violation Test ===")
    stale_hwnd = 0x7FFFFFFF
    stale_res = comp.execute({
        "action": "mouse_click",
        "x": 100,
        "y": 100,
        "expected_hwnd": stale_hwnd,
    })
    print("Stale action rejected:", not stale_res.success)
    print("Error message:", stale_res.error)
    assert not stale_res.success, "Expected stale target action to be aborted"
    assert "Stale target safety violation" in stale_res.error

    print("\nALL REAL-WINDOWS UIA E2E CHECKS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
