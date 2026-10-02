import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.load(r)


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


states = get("/health")["states"]
assert set(states) == {"delhi", "odisha", "bihar"}, f"expected 3 states, got {states}"
for s in states:
    opts = get(f"/options/{s}")
    case = {"state": s, **{f: v[0] for f, v in opts["fields"].items()}}
    if opts["uses_court_filings_90d"]:
        case["court_filings_90d"] = 500
    out = post("/predict", case)
    assert out["expected_days"] > 0, out
    assert out["risk_tier"] in ("Low", "Medium", "High"), out
    assert len(out["top_contributing_factors"]) == 5, out
    print(f"{s}: {out['expected_days']} days, tier {out['risk_tier']}, model {out['model_version']}")
print("smoke test passed")
