"""Cut the example storms of the README animations out of the storm-motion data set.

For each storm: the hourly rain, the storm core and the rain-mass track of the analysed hours, in a window of
1,400 km x 1,100 km around the track. The rain is stored without loss (multiples of 1/16 mm/h). The files written here
are in ``docs/animations/data``; this script is only needed to make them again or to add a storm.

    python docs/animations/extract_from_dataset.py <data set folder> <storm_uid> [<storm_uid> ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

OUT = Path(__file__).resolve().parent / "data"
HALF = (700.0, 550.0)                      # half size of the window kept (km, east and north)


def main(dataset, uids):
    dataset = Path(dataset)
    storms = pd.read_parquet(dataset / "storms" / "storms.parquet").set_index("storm_uid")
    tracks = pd.read_parquet(dataset / "storms" / "tracks.parquet")
    for uid in uids:
        s = storms.loc[uid]
        t = tracks[(tracks.storm_uid == uid) & tracks.analysed].sort_values("time")
        f = dataset / "rainfall" / (f"rain_{s.tile}_added.nc" if bool(s.added) else f"rain_{s.tile}.nc")
        with xr.open_dataset(f) as ds:
            d = ds.isel(storm=int(np.flatnonzero(ds.storm_uid.values == uid)[0]))
            hrs = t.hour_index.astype(int).values
            lat, lon = d.lat.values, d.lon.values
            lat0, lon0 = float(t.lat.mean()), float(t.lon.mean())
            jj = np.flatnonzero(np.abs((lat - lat0) * 111.32) <= HALF[1])
            ii = np.flatnonzero(np.abs((lon - lon0) * 111.32 * np.cos(np.radians(lat0))) <= HALF[0])
            rain = np.nan_to_num(d.rain.isel(hour=hrs, y=jj, x=ii).values.astype("float32"))
            mask = d.storm_mask.isel(hour=hrs, y=jj, x=ii).values == 1
        assert np.abs(rain - np.round(rain * 16) / 16).max() == 0
        np.savez_compressed(OUT / f"{uid}.npz", rain16=np.round(rain * 16).astype("uint16"),
                            mask=np.packbits(mask, axis=None), shape=np.array(mask.shape), lat=lat[jj], lon=lon[ii],
                            time=t.time.values.astype("datetime64[h]").astype(str),
                            track_lon=t.lon.values, track_lat=t.lat.values,
                            pattern_u=float(s.pattern_u_ms), pattern_v=float(s.pattern_v_ms),
                            linger_class=str(s.linger_class), tile=str(s.tile))
        print(uid, rain.shape, f"{(OUT / f'{uid}.npz').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
