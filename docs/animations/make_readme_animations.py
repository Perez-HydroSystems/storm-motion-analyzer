"""Make the two animations of the README from the example storms in ``docs/animations/data``.

    python docs/animations/make_readme_animations.py

tracking_steps.gif         the tracking of one storm in three steps: the rain, the identified storm with its
                           ellipse and centroid, and the track with the motion properties
storm_motion_classes.gif   four storms at the same scale: one that travels (coherent) and three that linger
                           (stall, turn, meander)
two_motions.gif            one storm with its two motions: the rain mass and the rain pattern
long_track.gif             one storm followed for a week in a single run (Hurricane Harvey, 153 hours)

The storms are extreme precipitation events of the storm-motion study of the contiguous United States (AORC hourly
precipitation), tracked on the tile enlarged by 15 degrees with the linking preset ``partition_keepsplit``; the 12 hours
shown are the hours with most storm rainfall over the tile (``analysis_hours="tile"``):
    A4_073   27 June 2019        travel        D7_001   29 August 2017   stall (Hurricane Harvey)
    B7_098   20 June 2016        turn          C9_264   20 July 2023     meander
    C7_337   9 June 2008         two motions (mesoscale convective system)
    A7_111   8 August 2020       tracking in three steps (convective system, northern Plains)
The rain-pattern velocities of the second animation are computed here with ``motion.pattern.storm_pattern``. The files
in ``docs/animations/data`` are made with ``extract_examples.py``, and the one of the long track (``harvey_2017``, tracked
from 25 August 2017 with storms identified from 2 mm/h cores) with ``extract_long_event.py``.
"""
from pathlib import Path

import numpy as np

from stormcatalog_analyzer.motion.pattern import storm_pattern
from stormcatalog_analyzer.plotting.motion_animation import (animate_storm_motion, animate_tracking_steps,
                                                             animate_two_motions)

HERE = Path(__file__).resolve().parent


def load(uid: str, label: str = "", sublabel: str = None) -> dict:
    z = np.load(HERE / "data" / f"{uid}.npz")
    shape = tuple(z["shape"])
    return dict(rain=z["rain16"].astype("float32") / 16, lon=z["lon"], lat=z["lat"], times=z["time"],
                mask=np.unpackbits(z["mask"])[:int(np.prod(shape))].reshape(shape).astype(bool),
                track_lon=z["track_lon"], track_lat=z["track_lat"], label=label, sublabel=sublabel,
                reference=(float(z["pattern_u"]), float(z["pattern_v"])))


def main():
    storms = [load("A4_073", "Travel", "keeps one direction of motion"),
              load("D7_001", "Stall", "stays over the same area"),
              load("B7_098", "Turn", "changes its heading"),
              load("C9_264", "Meander", "moves without a steady direction")]
    print(animate_storm_motion(
        storms, HERE / "storm_motion_classes.gif", title="Storm motion: travel, stall, turn, and meander",
        subtitle="Hourly rainfall and the track of the rain mass of four storms, 12 hours each. "
                 "All panels have the same scale."))

    s = load("C7_337", "Rain and tracks")
    hours = np.arange(s["rain"].shape[0])
    out, steps = storm_pattern(s["rain"], s["lon"], s["lat"], list(s["mask"]), hours, s["track_lon"], s["track_lat"],
                               seed=[20261103, 1], cross_check=False)
    # the window stored for the animation cuts the edges of this large storm, so the velocity computed here differs a
    # little from the one of the whole field (stored in the file), which is the one shown
    assert np.allclose((out["pattern_u"], out["pattern_v"]), s["reference"], atol=0.5), (out, s["reference"])
    s.update(step_u=steps["u"].values, step_v=steps["v"].values, step_ok=steps["accepted"].values,
             pattern_u=s["reference"][0], pattern_v=s["reference"][1])
    print(animate_two_motions(
        s, HERE / "two_motions.gif", half_width_km=900.0, half_height_km=680.0,
        title="Two motions of one storm: rain mass and rain pattern",
        subtitle="The rain mass follows the centroid of the rainfall. "
                 "The rain pattern follows the displacement of the rain features."))


def tracking_steps():
    z = np.load(HERE / "data" / "A7_111.npz")
    s = load("A7_111")
    s.update(direction_deg=float(z["direction_deg"]),
             properties={"Direction": f"{float(z['direction_deg']) % 360:.0f}° counter-clockwise from east",
                         "Path speed": f"{float(z['path_speed_ms']):.1f} m/s",
                         "Net speed": f"{float(z['net_speed_ms']):.1f} m/s",
                         "Mean rain rate": f"{float(z['mean_rate_mmh']):.1f} mm/h",
                         "Angular variance": f"{float(z['angular_variance']):.2f}"})
    print(animate_tracking_steps(
        s, HERE / "tracking_steps.gif",
        subtitle="A convective system over the northern Plains, 8 August 2020. Hourly rainfall, 12 hours."))


def long_track():
    z = np.load(HERE / "data" / "harvey_2017.npz")
    shape, n, step = tuple(z["shape"]), z["track_lon"].size, int(z["frame_step"])
    frame = np.minimum(np.arange(n) // step, shape[0] - 1)      # the rain is stored for the hours that are drawn
    frame[-1] = shape[0] - 1
    s = dict(rain=(z["rain16"].astype("float32") / 16)[frame], lon=z["lon"], lat=z["lat"], times=z["time"],
             mask=np.unpackbits(z["mask"])[:int(np.prod(shape))].reshape(shape).astype(bool)[frame],
             track_lon=z["track_lon"], track_lat=z["track_lat"], ref_lon=z["ref_lon"], ref_lat=z["ref_lat"],
             label="Hurricane Harvey", sublabel=f"{n} hours in one run", 
             ref_label="Pressure centre of the cyclone, not a rain track (NOAA HURDAT2, 6-hourly, interpolated)")
    print(animate_storm_motion(
        [s], HERE / "long_track.gif", ncols=1, half_width_km=800.0, half_height_km=600.0, fps=8.0, hold_frames=24,
        panel_width_in=7.6, frame_step=step, title="A storm tracked for a week: Hurricane Harvey, August 2017",
        subtitle="Hourly rainfall, the storm core and the track of its rain mass."))


if __name__ == "__main__":
    tracking_steps()
    main()
    long_track()
