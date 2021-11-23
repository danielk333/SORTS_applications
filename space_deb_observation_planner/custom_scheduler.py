import sorts

import numpy as np


class ObservedScanning(
            sorts.scheduler.StaticList,
            sorts.scheduler.ObservedParameters,
        ):
    pass


def get_scheduler():
    radar = sorts.radars.eiscat_uhf
    radar.rx = radar.rx[:1]

    end_t = 48.0*3600.0
    scan = sorts.radar.scans.Beampark(
        azimuth=90.0, 
        elevation=75.0,
        dwell=0.1,
    )

    scanner = sorts.controller.Scanner(
        radar,
        scan,
        t = np.arange(0, end_t, scan.dwell()*10),
        r = np.array([500e3]),
        t_slice = scan.dwell(),
    )

    return ObservedScanning(
        radar = radar, 
        controllers = [scanner], 
    )
