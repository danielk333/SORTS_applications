import sorts

import numpy as np

radar = sorts.radars.eiscat_uhf


class ObservedScanning(
            sorts.scheduler.StaticList,
            sorts.scheduler.ObservedParameters,
        ):
    pass


def get_scheduler():
    end_t = 48.0*3600.0
    scan = sorts.radar.scans.Beampark(
        azimuth=90.0, 
        elevation=75.0,
        dwell=0.1,
    )

    scanner = sorts.controller.Scanner(radar, scan)
    scanner.t = np.arange(0, end_t, scan.dwell())

    return ObservedScanning(
        radar = radar, 
        controllers = [scanner], 
    )
