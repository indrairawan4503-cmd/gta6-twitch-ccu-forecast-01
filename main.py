"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Variant A: an empirical comparable-shape model.

Idea
----
Every past big launch has a *shape*: how its Twitch category's hourly
viewers move through release day relative to the day's first hour, once
the game is actually live in the big markets.  For each comparable launch
we take the log-ratio

    r_h = ln(viewers_h / viewers_1)          for h = 2 .. 24

and, per hour, regress r_h on ln(viewers_1) across launches (weighted
least squares).  A launch whose first hour is already huge tends to be
near its peak already, so its later ratios are flatter; a launch whose
first hour is small is usually still climbing into its unlock waves, so
its later ratios run higher.  The regression slope captures exactly
that, and it is shrunk 20% toward zero so the forecast never leans on
an extrapolated slope for an hour-1 print outside the comparables'
range (GTA VI's first hour may be the biggest ever seen).

The forecast for hour h is then

    q_tau(h) = p1 * exp(median_h + z_tau * s_h)

where p1 is the case's hour-1 print, median_h is the shrunk regression
prediction at ln(p1), z_tau the standard normal quantile at the scored
level, and s_h the weighted residual standard deviation of the
regression (the honest cross-launch spread at that hour, floored at
0.05 log units).

Which launches are fitted
--------------------------
Only launches whose shape is about the day itself, not about an unlock
that had not happened yet:

* the target game's own pre-release category history is not a launch
  and is excluded;
* a launch is kept only when its hour-1 viewers are at least 10% of
  its own release-day peak.  Launches below that (a simultaneous global
  unlock late in the UTC day, e.g. 15:00Z or 18:00Z) spend the day
  exploding from near zero, which is not the shape of a rolling
  midnight-local unlock like GTA VI's;
* console-only launches (no PC in their platforms) get weight 1.0,
  launches with a PC version weight 0.4: GTA VI is console-only and
  unlocks at midnight local time, so the console-only rolling launches
  are the closest shape analogues.  PC-simultaneous launches are
  downweighted, not dropped, because they still carry the evening
  prime-time shape.

The case's own pre-release history (the GTA VI category's hourly
points) is read only as a scale sanity signal and never drives the
forecast: I examined it while building this entry — a quiet baseline
near ~1.4k viewers with a ~37k premiere spike — and it carries no
information about launch-day scale, so the forecast anchors on the
hour-1 print alone.  If the hour-1 print is missing we fall back to 1,
as the rules prescribe.

Robustness
----------
* Standard library only, no network, deterministic (comparables are
  processed in name order; no randomness, no wall clock).
* Every *.in in the input directory is read, one *.out written per
  case.  A case with no usable comparables, or an hour with fewer than
  four fitted launches, falls back to the house baseline's widening
  spread around the hour-1 print, so a forecast is always produced.
* All quantiles are clamped to (0, 1e9], non-decreasing, and rounded
  to 2 decimals half-up; the output stays far below the 65,536-byte
  cap and each run takes well under a second.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# A launch joins the fit only if its hour-1 print is at least this
# fraction of its own release-day peak (see module docstring).
MIN_H1_FRACTION = 0.10
# Weight of a launch that also shipped on PC; console-only is 1.0.
PC_WEIGHT = 0.4
# Shrinkage of the log(h1) regression slope toward zero.
BETA_SHRINK = 0.8
# Floor on the per-hour residual spread, in log units.
MIN_SPREAD = 0.05
# Guard rails on the median log-ratio (0.20x .. 4.95x the hour-1 print).
MEDIAN_LOG_MIN = -1.6
MEDIAN_LOG_MAX = 1.6
# House baseline spread used only as a fallback (0.15 per sqrt hour).
FALLBACK_SIGMA = 0.15
# A launch needs at least this many release-day hourly points to join.
MIN_DAY_HOURS = 12
# An hour needs at least this many fitted launches for the regression.
MIN_FIT_POINTS = 4


def _parse_time(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _norm_cdf_inv(p):
    """Inverse standard normal CDF (Acklam's approximation, ~1e-9)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


Z_BY_LEVEL = {t: _norm_cdf_inv(t) for t in LEVELS}


def release_day_hours(comp):
    """Map {hour 1..24: viewers} for a comparable's release (UTC) day."""
    try:
        day_start = _parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    hours = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        try:
            at = _parse_time(point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        offset = (at - day_start).total_seconds() / 3600.0
        if 0 <= offset < 24 and offset == int(offset):
            hours[int(offset) + 1] = float(viewers)
    return hours


def is_console_only(comp):
    """True when the launch shipped on consoles with no PC version."""
    platforms = (comp.get("platforms") or "").lower()
    return "pc" not in platforms


def collect_launches(case):
    """Return the fitted launches: (log h1, weight, {hour: viewers}).

    Each launch is kept only if its hour-1 print is a real share of its
    own release-day peak, so late-unlock explosions do not set the
    shape.  The target game's own pre-release history is excluded.
    """
    target_name = ((case.get("game") or {}).get("name") or "").strip()
    launches = []
    comparables = sorted(
        case.get("comparables") or [],
        key=lambda c: (c.get("name") or ""),
    )
    for comp in comparables:
        name = comp.get("name") or ""
        if target_name and name.startswith(target_name):
            continue  # the target's own category history, not a launch
        day = release_day_hours(comp)
        if 1 not in day or len(day) < MIN_DAY_HOURS:
            continue
        peak = max(day.values())
        if day[1] < MIN_H1_FRACTION * peak:
            continue  # unlock had not happened yet at hour 1
        weight = 1.0 if is_console_only(comp) else PC_WEIGHT
        launches.append((math.log(day[1]), weight, day))
    return launches


def _z_values(levels):
    """Standard normal quantile per level (precomputed for LEVELS)."""
    return [Z_BY_LEVEL.get(t, _norm_cdf_inv(t)) for t in levels]


def fallback_row(p1, hour, levels):
    """House-style widening spread around the hour-1 print."""
    return [round2(p1 * math.exp(zi * FALLBACK_SIGMA * math.sqrt(hour - 1)))
            for zi in _z_values(levels)]


def forecast_hour(launches, log_p1, hour, levels):
    """One hour's quantiles from the shrunk log(h1) regression.

    Returns unrounded *ratios* to the hour-1 print; the caller scales
    by p1 and rounds, so rounding never eats precision on a big print.
    """
    points = []
    for log_h1, weight, day in launches:
        if hour in day:
            points.append((log_h1, math.log(day[hour] / day[1]), weight))
    if len(points) < MIN_FIT_POINTS:
        return None
    total_w = sum(w for _, _, w in points)
    mean_x = sum(x * w for x, _, w in points) / total_w
    mean_y = sum(y * w for _, y, w in points) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for x, _, w in points)
    beta = 0.0
    if sxx > 0:
        beta = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in points) / sxx
    beta *= BETA_SHRINK
    median = mean_y + beta * (log_p1 - mean_x)
    median = max(MEDIAN_LOG_MIN, min(MEDIAN_LOG_MAX, median))
    var = sum(w * (y - (mean_y + beta * (x - mean_x))) ** 2
              for x, y, w in points) / total_w
    spread = max(MIN_SPREAD, math.sqrt(max(var, 0.0)))
    return [math.exp(median + zi * spread) for zi in _z_values(levels)]


def forecast_case(case, name):
    levels = case.get("levels") or LEVELS
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    p1 = 1.0
    if first is not None:
        try:
            p1 = max(1.0, float(first.get("viewers") or 1))
        except (TypeError, ValueError):
            p1 = 1.0
    launches = collect_launches(case)
    log_p1 = math.log(p1)
    rows = []
    for row in case.get("forecast") or []:
        hour = row["hour"]
        ratios = forecast_hour(launches, log_p1, hour, levels)
        if ratios is None:
            quantiles = fallback_row(p1, hour, levels)
        else:
            quantiles = [round2(max(1.0, min(1e9, p1 * q))) for q in ratios]
            quantiles = sorted(quantiles)
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
