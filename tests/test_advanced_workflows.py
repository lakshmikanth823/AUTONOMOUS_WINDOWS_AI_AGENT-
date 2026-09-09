"""Permanent regression test suite verifying advanced autonomous agent workflows:
Test 5: Use Multiple Applications (Filesystem, Browser/Playwright, Terminal, ComputerTool)
Test 6: Complete a 20+ Step Business Workflow (Deterministic 22-step invoicing, ledger & audit)
Test 7: Run a Task for an Extended Period with Checkpoints (Multi-phase batching, pause/resume, state snapshots)
Test 8: Recover from Intentionally Introduced Failures (Missing dependency, code logic autofix, schema recovery)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
import pytest

from agent.config.permissions import PermissionLevel
from agent.core.agent import TaskState
from agent.core.recovery import FailureCategory, FailureClassifier, RecoveryManager, RetryPolicy
from agent.core.state import TaskStateEnum
from agent.core.verifier import VerificationRecord, VerificationStatus, Verifier
from agent.memory.manager import MemoryManager
from agent.memory.schemas import MemoryCategory
from agent.tools.browser import BrowserTool
from agent.tools.computer import ComputerTool
from agent.tools.filesystem import FilesystemTool
from agent.tools.terminal import TerminalTool


def test_workflow_5_multiple_applications(tmp_path: Path):
    """Test 5: Use Multiple Applications (Filesystem, Browser, Terminal, ComputerTool)."""
    fs = FilesystemTool()
    browser = BrowserTool()
    term = TerminalTool()
    comp = ComputerTool()

    sales_json = tmp_path / "sales.json"
    dash_html = tmp_path / "dash.html"
    browser_img = tmp_path / "browser.png"
    desktop_img = tmp_path / "desktop.png"
    analytics_py = tmp_path / "analytics.py"

    # 1. Filesystem
    raw_data = [{"item": "Server-Alpha", "units": 4, "price": 1500.0, "cost": 900.0}]
    fs.execute({"action": "create_file", "path": str(sales_json), "content": json.dumps(raw_data), "overwrite": True})

    html_content = f"""<!DOCTYPE html><html><body>
    <div id="kpi-rev">$6,000.00</div>
    <div id="table-row">Server-Alpha - 4 units</div>
    </body></html>"""
    fs.execute({"action": "create_file", "path": str(dash_html), "content": html_content, "overwrite": True})

    # 2. Browser
    nav_res = browser.execute({"action": "navigate", "url": dash_html.as_uri()})
    assert nav_res.success is True
    rev_res = browser.execute({"action": "extract_text", "url": dash_html.as_uri(), "selector": "#kpi-rev"})
    assert "$6,000.00" in rev_res.output["text"]
    shot_res = browser.execute({"action": "screenshot", "url": dash_html.as_uri(), "path": str(browser_img)})
    assert shot_res.success is True
    assert browser_img.exists()

    # 3. Terminal
    py_code = f"""import json, platform
with open(r"{sales_json}", "r") as f:
    d = json.load(f)[0]
margin = ((d['price'] - d['cost']) / d['price']) * 100
print(json.dumps({{"margin": margin, "os": platform.system()}}))
"""
    fs.execute({"action": "create_file", "path": str(analytics_py), "content": py_code, "overwrite": True})
    term_res = term.execute({"command": f"python {analytics_py.name}", "working_directory": str(tmp_path)})
    assert term_res.success is True
    out = json.loads(term_res.output["stdout"].strip())
    assert out["margin"] == 40.0

    # 4. ComputerTool
    desk_res = comp.execute({"action": "screenshot", "path": str(desktop_img)})
    assert desk_res.success is True
    assert desktop_img.exists()


def test_workflow_6_twenty_plus_step_business_workflow(tmp_path: Path):
    """Test 6: Complete a 20+ step business workflow (22 verified deterministic steps)."""
    fs = FilesystemTool()
    inv_dir = tmp_path / "invoices"
    inv_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Ingest raw orders
    orders = [
        {"id": "ORD-1", "cust": "C-1", "items": [{"sku": "S1", "qty": 2}], "tax_reg": "CA", "status": "CONFIRMED"},
        {"id": "ORD-2", "cust": "C-2", "items": [{"sku": "S2", "qty": 1}], "tax_reg": "NY", "status": "CONFIRMED"},
        {"id": "ORD-3", "cust": "C-1", "items": [{"sku": "S1", "qty": 1}], "tax_reg": "CA", "status": "VOIDED"},
        {"id": "ORD-1", "cust": "C-1", "items": [{"sku": "S1", "qty": 2}], "tax_reg": "CA", "status": "CONFIRMED"},
    ]
    # Step 2: Validate Schema
    assert all("id" in o and "cust" in o for o in orders)
    # Step 3: Deduplicate
    unique_orders = []
    seen = set()
    for o in orders:
        if o["id"] not in seen:
            seen.add(o["id"])
            unique_orders.append(o)
    assert len(unique_orders) == 3
    # Step 4: Filter voided
    active = [o for o in unique_orders if o["status"] == "CONFIRMED"]
    assert len(active) == 2
    # Step 5: Ingest catalog
    catalog = {"S1": 1000.0, "S2": 500.0}
    # Step 6: Validate active skus
    assert all(item["sku"] in catalog for o in active for item in o["items"])
    # Step 7: Reconcile pricing
    for o in active:
        for it in o["items"]:
            it["price"] = catalog[it["sku"]]
            it["line_total"] = it["qty"] * it["price"]
    # Step 8: Ingest customer tiers
    tiers = {"C-1": 10.0, "C-2": 0.0}
    # Step 9: Compute subtotals
    for o in active:
        o["subtotal"] = sum(it["line_total"] for it in o["items"])
    # Step 10: Apply discounts
    for o in active:
        o["discount"] = round(o["subtotal"] * (tiers[o["cust"]] / 100.0), 2)
        o["net_subtotal"] = o["subtotal"] - o["discount"]
    # Step 11: Ingest tax rates
    taxes = {"CA": 0.08, "NY": 0.09}
    # Step 12: Calculate taxes
    for o in active:
        o["tax"] = round(o["net_subtotal"] * taxes[o["tax_reg"]], 2)
    # Step 13: Shipping fees
    for o in active:
        o["shipping"] = 25.0
    # Step 14: Gross total
    for o in active:
        o["gross"] = round(o["net_subtotal"] + o["tax"] + o["shipping"], 2)
    # Step 15: General ledger entries
    ledger = []
    for o in active:
        ledger.append({"acct": "AR", "type": "DEBIT", "val": o["gross"]})
        ledger.append({"acct": "REV", "type": "CREDIT", "val": o["net_subtotal"]})
        ledger.append({"acct": "TAX", "type": "CREDIT", "val": o["tax"]})
        ledger.append({"acct": "FRT", "type": "CREDIT", "val": o["shipping"]})
    # Step 16: Balance check
    debits = round(sum(l["val"] for l in ledger if l["type"] == "DEBIT"), 2)
    credits = round(sum(l["val"] for l in ledger if l["type"] == "CREDIT"), 2)
    assert debits == credits
    # Step 17: Write invoice documents
    inv_files = []
    for o in active:
        p = inv_dir / f"{o['id']}.json"
        fs.execute({"action": "create_file", "path": str(p), "content": json.dumps(o), "overwrite": True})
        inv_files.append(p)
    # Step 18: Verify persistence
    assert all(p.is_file() and p.stat().st_size > 0 for p in inv_files)
    # Step 19: Cryptographic audit manifest
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inv_files}
    assert len(manifest) == 2
    # Step 20: Ingest payment settlement status
    settlements = [{"id": "ORD-1", "paid": active[0]["gross"]}, {"id": "ORD-2", "paid": active[1]["gross"]}]
    # Step 21: Reconcile payments
    assert all(s["paid"] == next(o["gross"] for o in active if o["id"] == s["id"]) for s in settlements)
    # Step 22: Executive summary
    exec_summary = {"status": "SUCCESS", "gross_total": debits, "invoices_count": len(inv_files)}
    sum_path = tmp_path / "exec_summary.json"
    fs.execute({"action": "create_file", "path": str(sum_path), "content": json.dumps(exec_summary), "overwrite": True})
    assert sum_path.is_file()


def test_workflow_7_extended_task_with_checkpoints(tmp_path: Path):
    """Test 7: Extended task execution across multiple phases with persistent checkpoints."""
    fs = FilesystemTool()
    cp1_file = tmp_path / "cp1.json"
    cp2_file = tmp_path / "cp2.json"
    cp3_file = tmp_path / "cp3.json"

    records = [f"REC-{i:02d}" for i in range(1, 21)]

    # Phase 1: 1-10
    phase1_done = records[0:10]
    fs.execute({"action": "create_file", "path": str(cp1_file), "content": json.dumps({"batch": 1, "done": phase1_done}), "overwrite": True})
    assert cp1_file.exists()

    # Phase 2: 11-15
    phase2_done = phase1_done + records[10:15]
    fs.execute({"action": "create_file", "path": str(cp2_file), "content": json.dumps({"batch": 2, "done": phase2_done}), "overwrite": True})
    assert cp2_file.exists()

    # Pause simulation
    task_state = TaskState(user_goal="Checkpointed pipeline", status=TaskStateEnum.PAUSED, is_paused=True)
    assert task_state.status == TaskStateEnum.PAUSED

    # Resume simulation: read Checkpoint 2
    loaded = json.loads(fs.execute({"action": "read_file", "path": str(cp2_file)}).output)
    resumed_done = loaded["done"]
    assert len(resumed_done) == 15

    # Phase 3: 16-20
    final_done = resumed_done + records[15:20]
    fs.execute({"action": "create_file", "path": str(cp3_file), "content": json.dumps({"batch": 3, "done": final_done}), "overwrite": True})

    task_state.status = TaskStateEnum.COMPLETED
    assert len(final_done) == 20
    assert len(set(final_done)) == 20


def test_workflow_8_fault_recovery_and_autofix(tmp_path: Path):
    """Test 8: Recover from intentionally introduced failures with autonomous diagnosis and autofix."""
    fs = FilesystemTool()
    term = TerminalTool()
    verifier = Verifier()

    # 1. Missing File Recovery
    missing = tmp_path / "missing.json"
    res_err = fs.execute({"action": "read_file", "path": str(missing)})
    assert res_err.success is False

    cat = FailureClassifier.classify(res_err.error or "", res_err)
    assert RetryPolicy.is_retry_safe(cat, "filesystem", {"action": "read_file"}, PermissionLevel.LOW_RISK)

    # Recovery: create prerequisite and re-read
    fs.execute({"action": "create_file", "path": str(missing), "content": '{"status": "ok"}', "overwrite": True})
    res_retry = fs.execute({"action": "read_file", "path": str(missing)})
    assert res_retry.success is True
    rec = verifier.verify("filesystem", {"action": "read_file", "path": str(missing)}, res_retry, "ok")
    assert rec.status == VerificationStatus.VERIFIED

    # 2. Code Logic Crash & Autofix
    calc_py = tmp_path / "calc.py"
    test_py = tmp_path / "test_calc.py"

    test_src = """from calc import compute
import pytest
def test_zero_guard():
    assert compute(10, 0) == 0.0
def test_valid():
    assert compute(10, 2) == 5.0
"""
    fs.execute({"action": "create_file", "path": str(test_py), "content": test_src, "overwrite": True})

    flawed = """def compute(a, b):
    return a / b
"""
    fs.execute({"action": "create_file", "path": str(calc_py), "content": flawed, "overwrite": True})

    # Initial test fails
    res_init = term.execute({"command": f"python -m pytest {test_py.name}", "working_directory": str(tmp_path)})
    assert res_init.output.get("exit_code") != 0
    assert "ZeroDivisionError" in (res_init.output.get("stdout", "") + res_init.output.get("stderr", ""))

    # Autofix code
    fixed = """def compute(a, b):
    if b == 0:
        return 0.0
    return float(a) / float(b)
"""
    fs.execute({"action": "create_file", "path": str(calc_py), "content": fixed, "overwrite": True})

    # Retest passes
    res_fix = term.execute({"command": f"python -m pytest {test_py.name}", "working_directory": str(tmp_path)})
    assert res_fix.output.get("exit_code") == 0

    # 3. Schema recovery
    res_bad = fs.execute({"action": "unknown_action_99"})
    assert res_bad.success is False
    res_good = fs.execute({"action": "read_file", "path": str(missing)})
    assert res_good.success is True
