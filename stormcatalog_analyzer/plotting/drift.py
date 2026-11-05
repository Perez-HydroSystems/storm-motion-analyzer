"""Catalog-level envelope-drift diagnostic figure (``pipeline.generate_drift_diagnostics``)."""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt


def plot_drift_diagnostics(drift_df, save_path, domain_name, window_hr=72, axis_azimuth_deg=None,
                           highlight=None, dpi=300, font_size=0):
    """Two or three panels: (a) drift velocity vectors (vx, vy) of every event, (b) empirical CDF
    of the drift speed (whole event and wettest window), (c) if an axis is given, empirical CDF
    of the along-axis component. ``highlight`` = optional {label: speed_ms} reference speeds
    drawn as vertical lines (e.g., imposed design drifts). Returns the PNG path.
    """
    fs = font_size or 11
    has_axis = axis_azimuth_deg is not None and "drift_axis_ms" in drift_df
    n = 3 if has_axis else 2
    fig, ax = plt.subplots(1, n, figsize=(5.2 * n, 4.8))
    d = drift_df
    w = f"w{window_hr}_" if window_hr else None

    # (a) velocity vectors as points (east, north components)
    a = ax[0]
    a.scatter(d["drift_vx_ms"], d["drift_vy_ms"], s=10, color="0.3", label="whole event")
    if w and f"{w}drift_vx_ms" in d:
        a.scatter(d[f"{w}drift_vx_ms"], d[f"{w}drift_vy_ms"], s=10, facecolors="none", edgecolors="tab:blue",
                  label=f"wettest {window_hr} h")
    lim = np.nanpercentile(np.abs(np.r_[d["drift_vx_ms"], d["drift_vy_ms"]]), 99) * 1.3 + 0.1
    if w and f"{w}drift_vx_ms" in d:
        lim = max(lim, np.nanpercentile(np.abs(np.r_[d[f"{w}drift_vx_ms"], d[f"{w}drift_vy_ms"]]), 99) * 1.1)
    for r in (0.5, 1.0, 2.0):
        if r < lim:
            a.add_patch(plt.Circle((0, 0), r, fill=False, ls=":", color="0.6"))
            a.text(r * 0.71, r * 0.71, f"{r:g} m/s", fontsize=fs - 2, color="0.4")
    if has_axis:
        u = np.array([np.sin(np.deg2rad(axis_azimuth_deg)), np.cos(np.deg2rad(axis_azimuth_deg))])
        a.annotate("", xy=u * lim * 0.9, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.5))
        a.text(*(u * lim * 0.92), f"axis {axis_azimuth_deg:g}°", color="tab:red", fontsize=fs - 1)
    a.axhline(0, color="0.8", lw=0.8)
    a.axvline(0, color="0.8", lw=0.8)
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_aspect("equal")
    a.set_xlabel("Eastward drift (m/s)", fontsize=fs)
    a.set_ylabel("Northward drift (m/s)", fontsize=fs)
    a.set_title("(a) Envelope drift vectors", loc="left", fontweight="bold", fontsize=fs)
    a.legend(fontsize=fs - 2, loc="lower left")

    def ecdf(axh, col, label, color, ls="-"):
        v = np.sort(d[col].dropna().values)
        if v.size:
            axh.step(v, np.arange(1, v.size + 1) / v.size, where="post", color=color, ls=ls, label=label)
        return v

    # (b) speed ECDF
    b = ax[1]
    v = ecdf(b, "drift_speed_ms", "whole event", "0.2")
    if w and f"{w}drift_speed_ms" in d:
        ecdf(b, f"{w}drift_speed_ms", f"wettest {window_hr} h", "tab:blue", "--")
    for q in (0.5, 0.95):
        if v.size:
            b.axvline(np.quantile(v, q), color="0.6", ls=":", lw=0.8)
            b.text(np.quantile(v, q), 0.03 if q == 0.5 else 0.10, f" p{int(q * 100)} {np.quantile(v, q):.2f}", fontsize=fs - 2)
    for lab, sp in (highlight or {}).items():
        b.axvline(sp, color="tab:red", lw=1.2)
        b.text(sp, 0.5, f" {lab}", color="tab:red", fontsize=fs - 2, rotation=90, va="center")
    b.set_xlabel("Drift speed (m/s)", fontsize=fs)
    b.set_ylabel("Fraction of events", fontsize=fs)
    b.set_title("(b) Drift speed", loc="left", fontweight="bold", fontsize=fs)
    b.legend(fontsize=fs - 2, loc="lower right")
    b.grid(alpha=0.3)

    # (c) along-axis component
    if has_axis:
        c = ax[2]
        v = ecdf(c, "drift_axis_ms", "whole event", "0.2")
        if w and f"{w}drift_axis_ms" in d:
            ecdf(c, f"{w}drift_axis_ms", f"wettest {window_hr} h", "tab:blue", "--")
        c.axvline(0, color="0.7", lw=0.8)
        for lab, sp in (highlight or {}).items():
            c.axvline(sp, color="tab:red", lw=1.2)
        c.set_xlabel(f"Drift along the {axis_azimuth_deg:g}° axis (m/s)", fontsize=fs)
        c.set_ylabel("Fraction of events", fontsize=fs)
        c.set_title("(c) Along-axis drift", loc="left", fontweight="bold", fontsize=fs)
        c.legend(fontsize=fs - 2, loc="lower right")
        c.grid(alpha=0.3)

    fig.suptitle(f"Envelope drift of the rain field: {domain_name} (n = {len(d)})", fontsize=fs + 1)
    fig.tight_layout()
    out = os.path.join(save_path, f"storm_drift_diagnostic_{domain_name}.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
