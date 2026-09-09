"""Real-world E2E demonstration of Windows UI Automation (UIA) observation and semantic verification."""

import sys
sys.path.append('e:/AI_')

from agent.core.verifier import default_verifier, VerificationStatus
from agent.tools.computer import ComputerTool
from agent.tools.uia import UIAClient


def main() -> None:
    comp = ComputerTool()

    # ---------------------------------------------------------------------
    # 1. Observe active window controls via UIA
    # ---------------------------------------------------------------------
    print("--- 1. UIA Observation ---")
    obs_res = comp.execute({"action": "ui_elements", "max_elements": 30})
    print("ui_elements success:", obs_res.success)
    assert obs_res.success is True, f"ui_elements action failed: {obs_res.error}"

    out = obs_res.output
    win = out.get("window", {})
    clean_title = win.get("title", "").encode("ascii", "replace").decode("ascii")
    print("Active window title:", clean_title)
    print("Active window HWND:", win.get("hwnd"))
    print("Active window process:", win.get("process_name"))
    print("Discovered element count:", out.get("element_count"))
    assert out.get("element_count", 0) > 0, "Expected at least 1 UI element discovered"

    elements = out.get("elements", [])
    print("First 5 discovered controls:")
    for i, el in enumerate(elements[:5]):
        el_name = el.get("name", "").encode("ascii", "replace").decode("ascii")
        print(f"  [{i+1}] {el_name} | {el.get('control_type')} | center: {el.get('center')}")
        assert "name" in el
        assert "control_type" in el
        assert "center" in el
        assert len(el["center"]) == 2
        assert isinstance(el["center"][0], int)
        assert isinstance(el["center"][1], int)

    # ---------------------------------------------------------------------
    # 2. Semantic Verification of Observed State
    # ---------------------------------------------------------------------
    print("\n--- 2. Semantic State Verification ---")
    first_type = elements[0].get("control_type")
    first_name = elements[0].get("name", "")

    # A. Verify active window and element presence
    verif_args = {
        "action": "ui_elements",
        "expected_element_present": first_type,
        "expected_element_absent": "NonExistentDialog_87654321",
    }
    if clean_title:
        # Use first word of title for robust substring match
        title_word = clean_title.split()[0]
        verif_args["expected_window_active"] = title_word

    verif_record = default_verifier.verify("computer", verif_args, obs_res)
    print("Verification status:", verif_record.status)
    print("Verification details:", verif_record.verification.encode("ascii", "replace").decode("ascii"))
    assert verif_record.status == VerificationStatus.VERIFIED, "Semantic verification failed"

    # B. Verify that a non-existent element fails verification
    fail_record = default_verifier.verify(
        "computer",
        {"action": "ui_elements", "expected_element_present": "CompletelyImaginaryControlXYZ"},
        obs_res,
    )
    print("Negative assertion status (expected FAILED):", fail_record.status)
    assert fail_record.status == VerificationStatus.FAILED, "Negative assertion should fail"

    # ---------------------------------------------------------------------
    # 3. Dynamic Semantic Target Resolution Hook
    # ---------------------------------------------------------------------
    print("\n--- 3. Dynamic Semantic Target Resolution ---")
    client = UIAClient()
    resolved = client.find_element(first_type, hwnd=win.get("hwnd"))
    assert resolved is not None, f"Expected to resolve target element by type '{first_type}'"
    print(f"Resolved target '{first_type}' to center coordinates:", resolved.get("center"))

    # ---------------------------------------------------------------------
    # 4. Final Confirmation
    # ---------------------------------------------------------------------
    print("\nE2E UIA demonstration completed successfully.")


if __name__ == "__main__":
    main()
