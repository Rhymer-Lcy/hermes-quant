"""Build the free-source convertible-bond lake: universe + daily bars + premium series.

Resumable by design (scrapers fail): a bond already resolved -- served (parquet) or
missed (.miss marker with the error) -- is never re-requested, so re-running after a
network drop continues where it stopped. The JSL revision sample is pulled by
scripts/cb_double_low_study.py (a cross-check input, not part of the lake proper).
Coverage is reported against the bonds that could have been ALIVE inside the study
window -- a matured-in-2012 miss does not matter to a 2018-onward study.

    python scripts/build_cb_lake.py                  # resume: skip everything resolved
    python scripts/build_cb_lake.py --retry-misses   # forget recorded misses, try again
"""
import sys

from hermes.cb import data as cb

WINDOW_START = "2018-01-01"   # docs/cb_lake.md: the pre-registered study window


def main(retry_misses: bool = False) -> int:
    uni = cb.build_universe()
    print(f"universe: {len(uni)} bonds, listed {uni['listing_date'].min().date()} -> "
          f"{uni['listing_date'].max().date()}", flush=True)

    if retry_misses:
        for dir_ in (cb.BARS_DIR, cb.PREMIUM_DIR):
            for p in dir_.glob("*.miss"):
                p.unlink()

    codes = list(uni["code"])[::-1]   # newest first, so the study window fills in early
    for what, build, dir_ in [("bars", cb.build_bars, cb.BARS_DIR),
                              ("premium", cb.build_premium, cb.PREMIUM_DIR)]:
        build(codes)
        served = {p.stem for p in dir_.glob("*.parquet")}
        missed = {p.stem for p in dir_.glob("*.miss")}
        print(f"{what}: {len(served)}/{len(codes)} served, {len(missed)} misses", flush=True)

    both = ({p.stem for p in cb.BARS_DIR.glob("*.parquet")}
            & {p.stem for p in cb.PREMIUM_DIR.glob("*.parquet")})

    # A CB lives at most 6 years, so anything listed >= window start - 6y COULD reach the
    # window; bonds whose (served) bars end before the window are known dead and drop out.
    # A candidate with NO bars at all stays counted -- conservative, it may have matured
    # earlier, but "unresolved" must not silently read as "covered".
    bars = cb.load_bars()
    last = bars.groupby("code")["date"].max()
    horizon = str(int(WINDOW_START[:4]) - 6) + WINDOW_START[4:]
    candidates = set(uni.loc[uni["listing_date"] >= horizon, "code"])
    known_dead = set(last[last < WINDOW_START].index)
    in_window = candidates - known_dead
    missing_in_window = sorted(in_window - both)
    print(f"in-window bonds (could trade on/after {WINDOW_START}): {len(in_window)}, "
          f"of which missing from the lake: {len(missing_in_window)}")
    for code in missing_in_window:
        for dir_, what in [(cb.BARS_DIR, "bars"), (cb.PREMIUM_DIR, "premium")]:
            marker = dir_ / f"{code}.miss"
            if marker.exists():
                print(f"   IN-WINDOW MISS {code} [{what}]: {marker.read_text(encoding='utf-8')}")
        if not any((d / f"{code}.miss").exists() for d in (cb.BARS_DIR, cb.PREMIUM_DIR)):
            print(f"   IN-WINDOW MISS {code}: not yet pulled")
    return 1 if missing_in_window else 0


if __name__ == "__main__":
    sys.exit(main(retry_misses="--retry-misses" in sys.argv[1:]))
