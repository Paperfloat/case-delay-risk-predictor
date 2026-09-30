import pandas as pd
from common import get_args, load_config, features_name

args = get_args()
cfg = load_config(args)
df = pd.read_csv(f"data/processed/{features_name(cfg)}", dtype=str,
                 usecols=["event", "date_of_decision", "date_last_list_clean", "date_next_list"])


def dt(col):
    s = pd.to_datetime(df[col], errors="coerce")
    return s.where(s.dt.year.between(2005, 2025))


dec, last, nxt = dt("date_of_decision"), dt("date_last_list_clean"), dt("date_next_list")
pending = df["event"] == "0"
print(f"{cfg['state']}: rows {len(df)}, pending {pending.mean():.1%}")

monthly = dec.dropna().dt.to_period("M").value_counts().sort_index()
ratio = monthly / monthly.shift(1).rolling(3).mean()
low = ratio.nsmallest(5).index
print("\nSteepest falls in monthly decision volume (ratio to prior 3-month mean):")
print(pd.DataFrame({"decisions": monthly, "ratio": ratio.round(2)}).loc[low].sort_index().to_string())
print("\nDecisions per month, last 30 months present:")
print(monthly.tail(30).to_string())

pl = last[pending].dropna()
print("\nPending rows, last listing date quantiles:")
print(pl.quantile([0.5, 0.9, 0.99, 0.999]).to_string())
cand = pl.quantile(0.90).to_period("M").to_timestamp("M")
print(f"\nCandidate cutoff (end of month of 90th pct of pending last listing): {cand.date()}")
print("Pending rows with a next listing date:", round(nxt[pending].notna().mean(), 3))
gap = (cand - pl).dt.days / 365.25
print("Pending rows last listed >3 years before that candidate:", round((gap > 3).mean(), 3))
