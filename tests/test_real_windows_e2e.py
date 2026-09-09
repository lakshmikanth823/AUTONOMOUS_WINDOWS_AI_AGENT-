# Real‑world E2E demonstration of the new Windows computer actions

"""
This script runs a small end‑to‑end scenario using the
`ComputerTool` to exercise the newly added actions:

* ``write_clipboard`` – writes a known string to the Windows clipboard.
* ``region_screenshot`` – captures a rectangular region of the screen to a file.

The built‑in ``default_verifier`` validates the tool results.
"""

import sys
sys.path.append('e:/AI_')

import pathlib
import tempfile

from agent.tools.computer import ComputerTool
from agent.core.verifier import default_verifier


def main() -> None:
    comp = ComputerTool()

    # ---------------------------------------------------------------------
    # 1. Write to the clipboard and verify the result
    # ---------------------------------------------------------------------
    clipboard_text = "AGY clipboard test"
    write_res = comp.execute({"action": "write_clipboard", "text": clipboard_text})
    print("write_clipboard result:", write_res)
    write_ver = default_verifier.verify(
        "computer",
        {"action": "write_clipboard", "text": clipboard_text},
        write_res,
    )
    print("write_clipboard verification:", write_ver)

    # ---------------------------------------------------------------------
    # 2. Capture a screen region and verify the saved image
    # ---------------------------------------------------------------------
    # Use a temporary file so the script does not leave stray artefacts.
    screenshot_path = pathlib.Path(tempfile.gettempdir()) / "region_demo.png"
    region_res = comp.execute(
        {
            "action": "region_screenshot",
            "x": 0,
            "y": 0,
            "width": 200,
            "height": 200,
            "path": str(screenshot_path),
        }
    )
    print("region_screenshot result:", region_res)
    region_ver = default_verifier.verify(
        "computer",
        {
            "action": "region_screenshot",
            "path": str(screenshot_path),
        },
        region_res,
    )
    print("region_screenshot verification:", region_ver)

    # ---------------------------------------------------------------------
    # Final status
    # ---------------------------------------------------------------------
    if write_ver.passed and region_ver.passed:
        print("E2E demo completed successfully.")
    else:
        print("WARNING: Some verification steps failed.")


if __name__ == "__main__":
    main()
