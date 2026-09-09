"""Permanent test suite verifying the four fundamental autonomous agent workflows:
1. Create -> Execute -> Verify
2. Read -> Analyze -> Modify -> Test
3. Browse -> Research -> Save Structured Results
4. Research -> Code -> Test -> Fix -> Retest (Autofix across all edge cases)
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from agent.core.verifier import Verifier
from agent.tools.browser import BrowserTool
from agent.tools.filesystem import FilesystemTool
from agent.tools.terminal import TerminalTool


def test_workflow_1_create_execute_verify(tmp_path: Path):
    """Test 1: Create -> Execute -> Verify."""
    fs = FilesystemTool()
    term = TerminalTool()
    verifier = Verifier()

    script = tmp_path / "stats.py"
    output_json = tmp_path / "stats.json"

    # 1. Create
    code = """import json, sys
nums = [float(x) for x in sys.argv[1:]]
mean = sum(nums) / len(nums)
print(json.dumps({"mean": mean, "count": len(nums)}))
"""
    res_create = fs.execute({"action": "create_file", "path": str(script), "content": code, "overwrite": True})
    assert res_create.success is True

    # 2. Execute
    cmd = f"python {script.name} 10 20 30"
    res_exec = term.execute({"command": cmd, "working_directory": str(tmp_path)})
    assert res_exec.success is True
    assert res_exec.output.get("exit_code") == 0

    # 3. Verify
    data = json.loads(res_exec.output["stdout"].strip())
    assert data["mean"] == 20.0
    assert data["count"] == 3

    # Save artifact
    res_save = fs.execute({"action": "create_file", "path": str(output_json), "content": json.dumps(data), "overwrite": True})
    assert res_save.success is True
    assert output_json.is_file()


def test_workflow_2_read_analyze_modify_test(tmp_path: Path):
    """Test 2: Read -> Analyze -> Modify -> Test."""
    fs = FilesystemTool()
    term = TerminalTool()

    target_py = tmp_path / "discount.py"
    test_py = tmp_path / "test_discount.py"

    # Initial buggy implementation
    buggy = """def get_discount(price, pct):
    return price - (price * (pct / 100))
"""
    fs.execute({"action": "create_file", "path": str(target_py), "content": buggy, "overwrite": True})

    tests = """from discount import get_discount
import pytest

def test_valid():
    assert get_discount(100, 10) == 90.0

def test_negative_rejected():
    with pytest.raises(ValueError):
        get_discount(100, -5)
"""
    fs.execute({"action": "create_file", "path": str(test_py), "content": tests, "overwrite": True})

    # 1. Read
    read_res = fs.execute({"action": "read_file", "path": str(target_py)})
    assert "def get_discount" in read_res.output

    # 2. Analyze: initial test run fails
    init_res = term.execute({"command": f"python -m pytest {test_py.name}", "working_directory": str(tmp_path)})
    assert init_res.output.get("exit_code") != 0

    # 3. Modify: add validation
    fixed = """def get_discount(price, pct):
    if pct < 0 or pct > 100:
        raise ValueError("Invalid percentage")
    return round(price - (price * (pct / 100)), 2)
"""
    mod_res = fs.execute({"action": "create_file", "path": str(target_py), "content": fixed, "overwrite": True})
    assert mod_res.success is True

    # 4. Test: test run passes
    final_res = term.execute({"command": f"python -m pytest {test_py.name}", "working_directory": str(tmp_path)})
    assert final_res.output.get("exit_code") == 0


def test_workflow_3_browse_research_save_structured(tmp_path: Path):
    """Test 3: Browse -> Research -> Save Structured Results."""
    fs = FilesystemTool()
    browser = BrowserTool()

    html_file = tmp_path / "services.html"
    catalog_json = tmp_path / "catalog.json"

    html = """<!DOCTYPE html><html><body>
    <table id="svc">
        <tr><th>Name</th><th>Port</th><th>Status</th></tr>
        <tr><td>Auth</td><td>8080</td><td>UP</td></tr>
        <tr><td>DB</td><td>5432</td><td>UP</td></tr>
    </table>
</body></html>"""
    fs.execute({"action": "create_file", "path": str(html_file), "content": html, "overwrite": True})

    # 1. Browse
    nav_res = browser.execute({"action": "navigate", "url": html_file.as_uri()})
    assert nav_res.success is True

    # 2. Research (Extract)
    extract_res = browser.execute({"action": "extract_text", "url": html_file.as_uri(), "selector": "#svc"})
    assert "Auth" in extract_res.output["text"]
    assert "5432" in extract_res.output["text"]

    # Structure data
    items = [
        {"name": "Auth", "port": 8080, "status": "UP"},
        {"name": "DB", "port": 5432, "status": "UP"},
    ]

    # 3. Save Structured Results
    save_res = fs.execute({"action": "create_file", "path": str(catalog_json), "content": json.dumps(items, indent=2), "overwrite": True})
    assert save_res.success is True

    # Verify
    read_res = fs.execute({"action": "read_file", "path": str(catalog_json)})
    parsed = json.loads(read_res.output)
    assert len(parsed) == 2
    assert parsed[0]["name"] == "Auth"


def test_workflow_4_research_code_test_autofix_retest(tmp_path: Path):
    """Test 4: Research -> Code -> Test -> Fix -> Retest (Autofix)."""
    fs = FilesystemTool()
    term = TerminalTool()

    code_file = tmp_path / "cleaner.py"
    test_file = tmp_path / "test_cleaner.py"

    test_code = """from cleaner import clean_query
import pytest

def test_sorting():
    assert clean_query("b=2&a=1") == "a=1&b=2"

def test_deduplication():
    assert clean_query("x=1&x=1") == "x=1"

def test_empty():
    assert clean_query("") == ""
"""
    fs.execute({"action": "create_file", "path": str(test_file), "content": test_code, "overwrite": True})

    # 1. Initial flawed code (fails sorting and deduplication)
    flawed = """def clean_query(q):
    return q
"""
    fs.execute({"action": "create_file", "path": str(code_file), "content": flawed, "overwrite": True})

    # 2. Initial test fails
    test_1 = term.execute({"command": f"python -m pytest {test_file.name}", "working_directory": str(tmp_path)})
    assert test_1.output.get("exit_code") != 0

    # 3. Autofix
    fixed = """import urllib.parse

def clean_query(q):
    if not q:
        return ""
    parsed = urllib.parse.parse_qs(q, keep_blank_values=True)
    pairs = []
    for k in sorted(parsed.keys()):
        for v in sorted(list(set(parsed[k]))):
            pairs.append((k, v))
    return urllib.parse.urlencode(pairs)
"""
    fs.execute({"action": "create_file", "path": str(code_file), "content": fixed, "overwrite": True})

    # 4. Retest passes
    test_2 = term.execute({"command": f"python -m pytest {test_file.name}", "working_directory": str(tmp_path)})
    assert test_2.output.get("exit_code") == 0
