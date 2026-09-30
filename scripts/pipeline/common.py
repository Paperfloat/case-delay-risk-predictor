import argparse
from pathlib import Path
import yaml

CONFIG = Path(__file__).resolve().parents[2] / "config" / "states.yaml"


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="key in config/states.yaml")
    ap.add_argument("--pending", choices=["keep", "drop"],
                    help="override keep_pending from the config")
    ap.add_argument("--in-dir", default="data/processed")
    ap.add_argument("--out-dir", default=None)
    return ap.parse_args()


def load_config(args):
    raw = yaml.safe_load(open(CONFIG))
    if args.state not in raw["states"]:
        raise SystemExit(f"Unknown state '{args.state}'. Known: {list(raw['states'])}")
    c = dict(raw["states"][args.state])
    c["state"] = args.state
    c["slug"] = c.get("slug", args.state)
    c["years"] = c.get("years", raw["years"])
    if args.pending:
        c["keep_pending"] = args.pending == "keep"
    c["keep_pending"] = bool(c.get("keep_pending", False))
    c["tag"] = "_survival" if c["keep_pending"] else ""
    c["ext"] = "csv.gz" if c["keep_pending"] else "csv"
    c["span"] = f"{c['years'][0]}_{c['years'][-1]}"
    return c


def features_name(c):
    return f"cases_{c['span']}_{c['slug']}{c['tag']}_features.{c['ext']}"


def normalized_name(c):
    return f"cases_{c['span']}_{c['slug']}{c['tag']}_normalized.{c['ext']}"
