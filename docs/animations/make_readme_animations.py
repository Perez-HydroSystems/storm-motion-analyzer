"""Make the two animations of the README from the example storms in ``docs/animations/data``.

    python docs/animations/make_readme_animations.py

storm_motion_classes.gif   four storms at the same scale: one that travels (coherent) and three that linger
                           (stall, turn, meander)
two_motions.gif            one storm with its two motions: the rain mass and the rain pattern

The storms are extreme precipitation events of the storm-motion study of the contiguous United States (AORC hourly
precipitation), tracked on the tile enlarged by 5 degrees with the linking preset ``partition_keepsplit``:
    B7_139   6 September 2020    travel        D7_001   29 August 2017   stall
    C6_084   20-21 July 2009     turn          B8_238   24 July 2016     meander
    C8_140   7 June 2008         two motions
The rain-pattern velocities of the second animation are computed here with ``motion.pattern.storm_pattern``.
"""
from pathlib import Path

import numpy as np

from stormcatalog_analyzer.motion.pattern import storm_pattern
from stormcatalog_analyzer.plotting.motion_animation import animate_storm_motion, animate_two_motions

HERE = Path(__file__).resolve().parent


def load(uid: str, label: str = "", sublabel: str = None) -> dict:
    z = np.load(HERE / "data" / f"{uid}.npz")
    shape = tuple(z["shape"])
    return dict(rain=z["rain16"].astype("float32") / 16, lon=z["lon"], lat=z["lat"], times=z["time"],
                mask=np.unpackbits(z["mask"])[:int(np.prod(shape))].reshape(shape).astype(bool),
                track_lon=z["track_lon"], track_lat=z["track_lat"], label=label, sublabel=sublabel,
                reference=(float(z["pattern_u"]), float(z["pattern_v"])))


def main():
    storms = [load("B7_139", "Travel", "keeps one direction of motion"),
              load("D7_001", "Stall", "stays over the same area"),
              load("C6_084", "Turn", "changes its heading"),
              load("B8_238", "Meander", "moves without a steady direction")]
    print(animate_storm_motion(
        storms, HERE / "storm_motion_classes.gif", title="Storm motion: travel, stall, turn, and meander",
        subtitle="Hourly rainfall and the track of the rain mass of four storms, 12 hours each. "
                 "All panels have the same scale."))

    s = load("C8_140", "Rain and tracks")
    hours = np.arange(s["rain"].shape[0])
    out, steps = storm_pattern(s["rain"], s["lon"], s["lat"], list(s["mask"]), hours, s["track_lon"], s["track_lat"],
                               seed=[20261103, 1], cross_check=False)
    assert np.allclose((out["pattern_u"], out["pattern_v"]), s["reference"], atol=1e-6), (out, s["reference"])
    s.update(step_u=steps["u"].values, step_v=steps["v"].values, step_ok=steps["accepted"].values,
             pattern_u=out["pattern_u"], pattern_v=out["pattern_v"])
    print(animate_two_motions(
        s, HERE / "two_motions.gif", title="Two motions of one storm: rain mass and rain pattern",
        subtitle="The rain mass follows the centroid of the rainfall. "
                 "The rain pattern follows the displacement of the rain features."))


if __name__ == "__main__":
    main()
