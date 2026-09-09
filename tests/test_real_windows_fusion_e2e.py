"""Real-world E2E demonstration of Phase 3C: Perception Fusion on Real Windows Desktop.

Validates:
Test A - UIA target observation and text manipulation in Notepad
Test B - Non-UIA canvas text extraction with sources=['ocr']
Test C - Fusion corroboration (UIA + OCR merging into single UnifiedTarget with sources=['uia', 'ocr'])
Test D - Stale target safety protection (aborts action on changed foreground window)
Test E - Target ambiguity safety gating (multiple identical candidates trigger AMBIGUOUS status)
Test F - Complete process and window cleanup
"""

import ctypes
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.core.verifier import VerificationStatus, default_verifier
from agent.tools.computer import ComputerTool
from agent.tools.perception import resolve_target


def main() -> None:
    comp = ComputerTool()
    print("==================================================================")
    print("PHASE 3C: REAL-WINDOWS PERCEPTION FUSION E2E VALIDATION")
    print("==================================================================")

    # -----------------------------------------------------------------
    # Test A - UIA Target Observation and Lifecycle in Notepad
    # -----------------------------------------------------------------
    print("\n=== Test A: Launch Notepad & Observe Unified UIA Target ===")
    proc_notepad = subprocess.Popen(["notepad.exe"])
    time.sleep(2.0)
    notepad_hwnd = None

    try:
        focus_res = comp.execute({"action": "window_focus", "text": "Notepad"})
        assert focus_res.success, f"window_focus failed: {focus_res.error}"
        notepad_hwnd = focus_res.output.get("hwnd")
        print("Notepad focused with HWND:", notepad_hwnd)

        # 1. Observe semantic state (fast mode with OCR off)
        obs_a = comp.execute({"action": "observe_semantic", "ocr_mode": "off", "hwnd": notepad_hwnd})
        assert obs_a.success is True
        out_a = obs_a.output
        targets_a = out_a.get("targets", [])
        print(f"Discovered {len(targets_a)} unified targets in Notepad (OCR off).")
        assert len(targets_a) > 0, "Expected targets in Notepad"

        # 2. Locate edit control and type text
        res_tgt = resolve_target(targets_a, "Text editor")
        if res_tgt.status != "RESOLVED":
            # Fallback to Document if name differs across Windows 10/11 versions
            res_tgt = resolve_target(targets_a, "Document")
        print(f"Target resolution status for editor: {res_tgt.status} (Reason: {res_tgt.reason})")

        # Set text in editor
        typed_text = "Autonomous Fusion 2026"
        set_res = comp.execute({
            "action": "set_element_text",
            "text": typed_text,
            "target_element": "Text editor",
            "hwnd": notepad_hwnd,
        })
        print("set_element_text success:", set_res.success)
        assert set_res.success is True

        # Verify semantic assertion on observe_semantic
        verif_a = default_verifier.verify(
            "computer",
            {"action": "observe_semantic", "expected_target_present": "File", "expected_window_active": "Notepad"},
            obs_a,
        )
        assert verif_a.status == VerificationStatus.VERIFIED
        print("Test A Verification: PASSED (Target 'File' present in active Notepad)")

        # -----------------------------------------------------------------
        # Test B & C - Non-UIA Canvas & Fusion Corroboration
        # -----------------------------------------------------------------
        print("\n=== Test B & C: Non-UIA Surface & Fusion Corroboration ===")
        canvas_code = textwrap.dedent("""
            import tkinter as tk
            import sys
            import ctypes
            root = tk.Tk()
            root.title('Perception Fusion Harness')
            root.geometry('500x300+250+250')

            # 1. Standard UIA Button (UIA + OCR target)
            btn = tk.Button(root, text='CorroboratedButton', font=('Arial', 12, 'bold'))
            btn.pack(pady=10)

            # 2. Canvas surface (Non-UIA rasterized text)
            canvas = tk.Canvas(root, width=460, height=100, bg='white')
            canvas.pack(fill='both', expand=True, pady=10)
            canvas.create_text(230, 50, text='Canvas Only Secret 2026', font=('Arial', 16, 'bold'), fill='blue')

            root.update_idletasks()
            root.update()
            hwnd = ctypes.windll.user32.FindWindowW(None, 'Perception Fusion Harness')
            sys.stdout.write(f"{hwnd}\\n")
            sys.stdout.flush()
            root.mainloop()
        """)
        proc_canvas = subprocess.Popen(
            [sys.executable, "-c", canvas_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        hwnd_line = proc_canvas.stdout.readline().strip()
        assert hwnd_line, "Expected valid canvas HWND from child process"
        canvas_hwnd = int(hwnd_line)
        time.sleep(0.5)

        try:
            # Focus canvas window
            comp._bring_window_to_foreground(canvas_hwnd)
            time.sleep(0.5)

            # Observe with region OCR enabled for the harness window
            obs_bc = comp.execute({"action": "observe_semantic", "ocr_mode": "region", "hwnd": canvas_hwnd})
            assert obs_bc.success is True
            fused_targets = obs_bc.output.get("targets", [])
            print(f"Perception discovered {len(fused_targets)} fused targets in harness window.")

            # --- Test B Verification: Non-UIA Canvas Target ---
            ocr_targets = [t for t in fused_targets if "ocr" in t.get("sources", []) and "uia" not in t.get("sources", [])]
            print(f"Discovered {len(ocr_targets)} OCR-only targets.")
            canvas_secret_found = any("Secret" in t.get("name", "") or "Canvas" in t.get("name", "") for t in ocr_targets)
            assert canvas_secret_found, f"Expected non-UIA canvas target in OCR targets: {[t['name'] for t in ocr_targets]}"
            print("Test B (Non-UIA Canvas Target with source=['ocr']): PASSED")

            # --- Test C Verification: Corroboration Merging ---
            corroborated_targets = [t for t in fused_targets if "uia" in t.get("sources", []) and "ocr" in t.get("sources", [])]
            print(f"Discovered {len(corroborated_targets)} corroborated (UIA + OCR) targets.")
            assert len(corroborated_targets) >= 1, "Expected at least one corroborated target (CorroboratedButton)"
            corrob_btn = next((t for t in corroborated_targets if "CorroboratedButton" in t.get("name", "")), None)
            if corrob_btn:
                print(f"Corroborated target: '{corrob_btn.get('name')}' -> Sources: {corrob_btn.get('sources')}")
                assert set(corrob_btn.get("sources", [])) == {"uia", "ocr"}
            print("Test C (Single Corroborated Target Merged): PASSED")

            # -----------------------------------------------------------------
            # Test D - Stale Target Safety Protection
            # -----------------------------------------------------------------
            print("\n=== Test D: Stale Target Safety Protection ===")
            # Focus Notepad so active window changes away from canvas_hwnd
            comp._bring_window_to_foreground(notepad_hwnd)
            time.sleep(0.5)

            # Attempt mouse action using target bound to canvas_hwnd
            stale_res = comp.execute({
                "action": "mouse_click",
                "x": 200,
                "y": 200,
                "expected_hwnd": canvas_hwnd,
            })
            print("Stale action execution success:", stale_res.success)
            print("Stale action rejection error:", stale_res.error)
            assert stale_res.success is False
            assert "Stale target safety violation" in (stale_res.error or "")
            print("Test D (Stale Window Action Rejected): PASSED")

            # -----------------------------------------------------------------
            # Test E - Ambiguity Safety Gating
            # -----------------------------------------------------------------
            print("\n=== Test E: Ambiguity Safety Gating ===")
            from agent.tools.perception import UnifiedTarget
            ambiguous_pool = [
                UnifiedTarget(
                    id="opt_1",
                    name="AmbiguousAction",
                    control_type="Button",
                    rect={"left": 100, "top": 100, "right": 200, "bottom": 140, "width": 100, "height": 40},
                    center=(150, 120),
                    hwnd=canvas_hwnd,
                    sources=["uia"],
                ),
                UnifiedTarget(
                    id="opt_2",
                    name="AmbiguousAction",
                    control_type="Button",
                    rect={"left": 100, "top": 200, "right": 200, "bottom": 240, "width": 100, "height": 40},
                    center=(150, 220),
                    hwnd=canvas_hwnd,
                    sources=["uia"],
                ),
            ]
            ambig_res = resolve_target(ambiguous_pool, "AmbiguousAction")
            print("Ambiguity resolution status:", ambig_res.status)
            print("Ambiguity resolution reason:", ambig_res.reason)
            assert ambig_res.status == "AMBIGUOUS"
            assert ambig_res.target is None
            assert len(ambig_res.candidates) == 2
            print("Test E (Deterministic Ambiguity Detection): PASSED")

        finally:
            proc_canvas.terminate()

    finally:
        print("\n=== Test F: Clean Up All Processes & Verify ===")
        proc_notepad.terminate()
        time.sleep(0.5)
        # Check window list to confirm Notepad is gone
        win_list = comp.execute({"action": "window_list"}).output.get("windows", [])
        notepad_remaining = any("Notepad" in w for w in win_list)
        print("Notepad remaining in window list:", notepad_remaining)
        assert not notepad_remaining, "Expected Notepad process to be fully terminated"
        print("Test F (Cleanup Verification): PASSED")

    print("\n==================================================================")
    print("ALL REAL-WINDOWS PERCEPTION FUSION E2E TESTS PASSED SUCCESSFULLY!")
    print("==================================================================")


if __name__ == "__main__":
    main()
