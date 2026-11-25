"""Track one storm over several days in a single run, and store it for the long-track animation of the README.

The example is Hurricane Harvey: 168 hours from 25 August 2017 on a fixed area from Texas to Alabama (AORC hourly
precipitation). The steps are the ones of any long event:

    1. read the hourly rain of the whole period on a fixed area          (``enlarged.read_aorc_field``)
    2. track every rain area of the field, hour by hour                  (``enlarged.track_field``)
    3. choose the storm: the track with most rain inside a box           (``enlarged.track_table``,
       of interest, and its longest run of consecutive hours              ``enlarged.select_circle_storm``)
    4. describe its motion over all its hours                            (``enlarged.v6_descriptors``)

The storms are identified from cores of 2 mm/h grown into the rain area (``core_threshold_mmhr=2``). Over several
days a tropical cyclone joins and releases distant rain bands; with cores, these bands move the centroid much less
than with the whole rain area above 0.5 mm/h (largest hourly step 75 km in place of 270 km for this storm).

    python docs/animations/extract_long_event.py <AORC folder> [<best-track csv>]

The optional best-track file holds the positions of the centre of the hurricane (HURDAT2 of the National Hurricane
Center, https://www.nhc.noaa.gov/data/#hurdat, as a table with the columns tc_id, time, lat, lon). They are stored next
to the track, interpolated from the 6-hourly positions to the hours of the storm, and drawn as a reference: they are
the pressure centre of the cyclone, not a rain track. The file written here is
``docs/animations/data/harvey_2017.npz``; this script is only needed to make it again or to track another event.
"""
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from stormcatalog_analyzer import enlarged as E
from stormcatalog_analyzer.motion.linger import classify_track

OUT = Path(__file__).resolve().parent / "data"
NAME, TC_ID = "harvey_2017", "AL092017"
T0, T1 = "2017-08-25 01:00", "2017-09-01 00:00"              # 168 hours
AREA = (-106.0, 23.5, -82.0, 40.0)                           # west, south, east, north of the tracked field
BOX = (-98.5, 27.0, -92.5, 31.5)                             # where the storm is chosen (Texas and Louisiana coast)
CORE_MMHR = 2.0                                              # identification by cores
MIN_AREA_KM2 = 13600.0                                       # from the first to the last hour with a core above this
HALF = (850.0, 650.0)                                        # half size of the window kept (km, east and north)
FRAME_STEP = 2                                               # the rain is stored every 2 hours (the track, every hour)


def main(aorc, best_track=None):
    times = pd.date_range(T0, T1, freq="h").values
    fld = E.read_aorc_field(Path(aorc), times, *AREA)
    rain0 = np.nan_to_num(fld.rain, nan=0.0)
    cfg = E.EnlargedConfig(tracker="robust:partition_keepsplit", core_threshold_mmhr=CORE_MMHR)
    t = time.time()
    labels = E.track_field(rain0, cfg, fld.lat, fld.lon)
    seconds = time.time() - t
    area = E.pixel_area_km2(fld.lat, fld.lon)
    box = ((fld.lat[:, None] >= BOX[1]) & (fld.lat[:, None] <= BOX[3]) & (fld.lon[None, :] >= BOX[0])
           & (fld.lon[None, :] <= BOX[2])).astype(float)
    tab = E.track_table(labels, rain0, area, weight=box, lat=fld.lat, lon=fld.lon)
    k, run = E.select_circle_storm(tab)
    hours, _ = E.trim_hours(tab[tab.track_id == k], run, MIN_AREA_KM2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = E.v6_descriptors(rain0, labels, k, hours, fld.lat, fld.lon, fld.times, NAME,
                             replace(cfg, window_hr=len(hours)))        # all the hours, not a window of 12
    wh = d["_win_hours"]
    tlon, tlat = np.asarray(d["track_lon"]), np.asarray(d["track_lat"])
    core = float(np.mean([area[labels[h] == k].sum() for h in wh]))
    cls = classify_track(tlon, tlat, core, d["coherent"], d["net_length_km"], d["net_speed_ms"])
    print(f"field {rain0.shape}: tracking {seconds:.0f} s, {tab.track_id.nunique()} tracks; storm followed for "
          f"{len(wh)} hours ({d['start_time']} to {d['end_time']})")
    print(f"path {d['path_length_km']:.0f} km, net {d['net_length_km']:.0f} km, direction {d['direction_deg']:.0f} deg, "
          f"path speed {d['path_speed_ms']:.1f} m/s, net speed {d['net_speed_ms']:.1f} m/s, "
          f"coherent {d['coherent']}, class {cls['linger_class']}, largest step {d['max_step_km']:.0f} km")
    tt = pd.to_datetime(np.asarray(fld.times)[wh])
    ref_lon = ref_lat = np.full(len(wh), np.nan)
    if best_track:                                            # centre of the hurricane at the hours of the storm
        h = pd.read_csv(best_track, parse_dates=["time"])
        h = h[h.tc_id == TC_ID].sort_values("time")
        x = ((tt - h.time.iloc[0]) / pd.Timedelta(hours=1)).values
        hx = ((h.time - h.time.iloc[0]) / pd.Timedelta(hours=1)).values
        ref_lon = np.interp(x, hx, h.lon.values, left=np.nan, right=np.nan)
        ref_lat = np.interp(x, hx, h.lat.values, left=np.nan, right=np.nan)
    lat0, lon0 = 0.5 * (tlat.min() + tlat.max()), 0.5 * (tlon.min() + tlon.max())
    jj = np.flatnonzero(np.abs((fld.lat - lat0) * 111.32) <= HALF[1])
    ii = np.flatnonzero(np.abs((fld.lon - lon0) * 111.32 * np.cos(np.radians(lat0))) <= HALF[0])
    shown = sorted(set(range(0, len(wh), FRAME_STEP)) | {len(wh) - 1})   # the hours drawn in the animation
    rain = rain0[wh[shown]][:, jj][:, :, ii].astype("float32")
    mask = (labels[wh[shown]] == k)[:, jj][:, :, ii]
    assert np.abs(rain - np.round(rain * 16) / 16).max() == 0
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{NAME}.npz", rain16=np.round(rain * 16).astype("uint16"),
                        mask=np.packbits(mask, axis=None), shape=np.array(mask.shape), lat=fld.lat[jj], lon=fld.lon[ii],
                        time=tt.values.astype("datetime64[h]").astype(str), track_lon=tlon, track_lat=tlat,
                        frame_hours=np.array(shown), frame_step=FRAME_STEP,
                        ref_lon=ref_lon, ref_lat=ref_lat, linger_class=str(cls["linger_class"]),
                        direction_deg=float(d["direction_deg"]), net_speed_ms=float(d["net_speed_ms"]),
                        path_speed_ms=float(d["path_speed_ms"]), coherent=bool(d["coherent"]))
    print(NAME, rain.shape, f"{(OUT / f'{NAME}.npz').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:3])
