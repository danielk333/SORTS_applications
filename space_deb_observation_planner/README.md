# SST Planner

## custom scheduler for observation

Example beampark scheduler for prediction of observation results

```python

import sorts
import numpy as np


class ObservedScanning(
            sorts.scheduler.StaticList,
            sorts.scheduler.ObservedParameters,
        ):
    pass


radar = sorts.radars.eiscat_uhf

end_t = 48.0*3600.0
scan = sorts.radar.scans.Beampark(
    azimuth=90.0, 
    elevation=75.0,
    dwell=0.1,
)

scanner = sorts.controller.Scanner(radar, scan)
scanner.t = np.arange(0, end_t, scan.dwell())

scheduler = ObservedScanning(
    radar = radar, 
    controllers = [scanner], 
)
```