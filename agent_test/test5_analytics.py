import json, platform, sys
with open(r"E:\AI_\agent_test\test5_sales_input.json", "r", encoding="utf-8") as f:
    data = json.load(f)

store_margins = {}
for d in data:
    rev = d["units"] * d["unit_price"]
    cost = d["units"] * d["cost"]
    margin_pct = round(((rev - cost) / rev) * 100, 2)
    store_margins[d["store"]] = {"revenue": rev, "profit": rev - cost, "margin_pct": margin_pct}

sys_info = {
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "cpu_arch": platform.machine()
}

output = {"margins": store_margins, "system_info": sys_info}
print(json.dumps(output))
