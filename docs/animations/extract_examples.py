"""Make the example storms of the README animations from a storm catalog event and the AORC precipitation.

For each storm: the event is tracked on the tile enlarged by 15 degrees (linking preset ``partition_keepsplit``, the
12 hours with most storm rainfall over the tile), and the hourly rain, the storm core and the rain-mass track of those
hours are stored in a window around the track. The rain is stored without loss (multiples of 1/16 mm/h). The files
written here are in ``docs/animations/data``; this script is only needed to make them again or to add a storm.

    python docs/animations/extract_examples.py <AORC folder> <catalog event file> <tile> <storm_uid> [--large]

``--large`` keeps a window of 2,000 km x 1,500 km (large systems) in place of 1,400 km x 1,100 km.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from stormcatalog_analyzer import enlarged as E

OUT = Path(__file__).resolve().parent / "data"
MARGIN = 15.0                                                # degrees added to the tile on every side
HALF = {False: (700.0, 550.0), True: (1000.0, 750.0)}        # half size of the window kept (km, east and north)


def main(aorc, event_file, tile, uid, large=False):
    ev = E.read_catalog_event(Path(event_file))
    cfg = E.EnlargedConfig(margin_deg=MARGIN, tracker="robust:partition_keepsplit", analysis_hours="tile", pattern=True,
                           linger_class=True)
    fld = E.read_aorc_field(Path(aorc), ev.times, *E.tile_bounds_with_margin(ev, MARGIN))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row, _ = E.analyze_event(ev, fld, cfg, tile=tile, with_tile=False, return_footprints=False, keep_private=True)
    labels = row.pop("_labels")
    rain0 = np.nan_to_num(fld.rain, nan=0.0)
    # the storm of the event: the one with most rain inside the screening circle
    r0, c0 = int(np.argmin(np.abs(fld.lat - ev.lat[0]))), int(np.argmin(np.abs(fld.lon - ev.lon[0])))
    circle = np.zeros(rain0.shape[1:])
    circle[r0:r0 + len(ev.lat), c0:c0 + len(ev.lon)] = ev.circle_on_tile()
    k, _ = E.select_circle_storm(E.track_table(labels, rain0, E.pixel_area_km2(fld.lat, fld.lon), circle))
    position = {pd.Timestamp(t): i for i, t in enumerate(pd.to_datetime(fld.times))}
    hrs = np.array([position[pd.Timestamp(t)] for t in row["big_track_time"]])
    tlon, tlat = np.asarray(row["big_track_lon"], float), np.asarray(row["big_track_lat"], float)
    half = HALF[bool(large)]
    lat0, lon0 = float(tlat.mean()), float(tlon.mean())
    jj = np.flatnonzero(np.abs((fld.lat - lat0) * 111.32) <= half[1])
    ii = np.flatnonzero(np.abs((fld.lon - lon0) * 111.32 * np.cos(np.radians(lat0))) <= half[0])
    rain = rain0[hrs][:, jj][:, :, ii].astype("float32")
    mask = (labels[hrs] == k)[:, jj][:, :, ii]
    assert np.abs(rain - np.round(rain * 16) / 16).max() == 0
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{uid}.npz", rain16=np.round(rain * 16).astype("uint16"),
                        mask=np.packbits(mask, axis=None), shape=np.array(mask.shape), lat=fld.lat[jj], lon=fld.lon[ii],
                        time=np.array(row["big_track_time"]).astype("datetime64[h]").astype(str),
                        track_lon=tlon, track_lat=tlat, pattern_u=float(row["pattern_u"]), pattern_v=float(row["pattern_v"]),
                        linger_class=str(row["linger_class"]), tile=str(tile), tile_bounds=np.array(ev.tile_bounds),
                        direction_deg=float(row["big_direction_deg"]),
                        pattern_direction_deg=float(row["pattern_direction_deg"]), pattern_speed=float(row["pattern_speed"]),
                        net_speed_ms=float(row["big_net_speed_ms"]), path_speed_ms=float(row["big_path_speed_ms"]),
                        angular_variance=float(row["big_angular_variance"]), mean_rate_mmh=float(row["big_avg_intensity_mmh"]))
    print(uid, rain.shape, str(row["linger_class"]), f"{(OUT / f'{uid}.npz').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--large"]
    main(*args[:4], large="--large" in sys.argv)
