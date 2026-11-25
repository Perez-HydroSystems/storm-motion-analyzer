"""The motion animations (plotting.motion_animation) write a looping GIF with one frame per hour."""
import numpy as np
import pytest
from scipy import ndimage

from stormcatalog_analyzer.plotting.motion_animation import animate_storm_motion, animate_two_motions

PIL = pytest.importorskip("PIL.Image")
LAT = 40 + np.arange(90) / 30.0
LON = -95 + np.arange(110) / 30.0


def storm(n=6, step=(2.0, 3.0), label="Travel"):
    yy, xx = np.indices((90, 110))
    base = 12 * np.exp(-(((yy - 30) / 9) ** 2 + ((xx - 30) / 12) ** 2))
    rain = np.array([ndimage.shift(base, (step[0] * t, step[1] * t), order=1) for t in range(n)])
    mask = rain > 0.5
    lon2d, lat2d = np.meshgrid(LON, LAT)
    w = np.where(mask, rain, 0.0)
    return dict(rain=rain, lon=LON, lat=LAT, mask=mask, label=label, sublabel="synthetic",
                track_lon=np.array([(lon2d * x).sum() / x.sum() for x in w]),
                track_lat=np.array([(lat2d * x).sum() / x.sum() for x in w]),
                times=np.arange("2020-01-01T00", f"2020-01-01T{n:02d}", dtype="datetime64[h]").astype(str))


def test_storm_motion_gif(tmp_path):
    p = animate_storm_motion([storm(6), storm(4, (0.2, 0.1), "Stall")], tmp_path / "a.gif", title="Two storms",
                             half_width_km=250, half_height_km=200, scale_km=100, borders=False, dpi=50)
    im = PIL.open(p)
    assert im.n_frames == 6 and im.info.get("loop") == 0


def test_two_motions_gif(tmp_path):
    s = storm(6)
    s.update(step_u=np.full(5, 8.0), step_v=np.full(5, 6.0), step_ok=np.array([True, True, False, True, True]))
    p = animate_two_motions(s, tmp_path / "b.gif", half_width_km=250, half_height_km=200, scale_km=100,
                            borders=False, dpi=50)
    assert PIL.open(p).n_frames == 6


def test_frame_step_keeps_the_last_hour(tmp_path):
    s = storm(6)
    s.update(ref_lon=s["track_lon"] + 0.1, ref_lat=s["track_lat"], ref_label="Reference")
    p = animate_storm_motion([s], tmp_path / "c.gif", ncols=1, half_width_km=250, half_height_km=200, scale_km=100,
                             borders=False, dpi=50, frame_step=4)
    assert PIL.open(p).n_frames == 3                      # hours 0 and 4, and the last hour
