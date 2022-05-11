#!/usr/bin/env python

import numpy as np
import sorts


class TrackingScheduler(
                sorts.scheduler.StaticList, 
                sorts.scheduler.ObservedParameters,
            ):

    def __init__(
                    self, radar, t, states, 
                    track_len=3600.0, profiler=None, logger=None, 
                    **kwargs
                ):
        self.passes = radar.find_passes(t, states)
        passes = sorts.passes.group_passes(self.passes)

        controllers = []
        for txi in range(len(passes)):
            for psi in range(len(passes[txi])):
                if len(passes[txi][psi]) == 0:
                    continue
                t_min = passes[txi][psi][0].start()
                t_max = passes[txi][psi][0].end()
                for ps in passes[txi][psi][1:]:
                    if ps.start() < t_min:
                        t_min = ps.start
                    if ps.end() > t_max:
                        t_max = ps.end()

                inds = np.logical_and(t >= t_min, t <= t_max)

                tracker = sorts.controller.Tracker(
                    radar = radar, 
                    t = t[inds], 
                    ecefs = states[:3, inds],
                    dwell = track_len,
                )
                controllers.append(tracker)

        super().__init__(
            radar=radar, 
            controllers=controllers,
            logger=logger, 
            profiler=profiler,
            **kwargs
        )
