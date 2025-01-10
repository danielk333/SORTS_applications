import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from astropy.time import Time, TimeDelta

import sorts


class ScanAndChase(sorts.scheduler.ObservedParameters):
    def __init__(self, radar, scan, epoch, profiler=None, logger=None, **kwargs):
        super().__init__(
            radar=radar,
            logger=logger,
            profiler=profiler,
        )
        self.end_time = kwargs.get("end_time", 3600.0 * 24.0)
        self.timeslice = kwargs.get("timeslice", 0.1)
        self.max_predict_time = kwargs.get("max_predict_time", 3600.0)
        self.epoch = epoch
        self.scan = scan
        self.tracking_object = None
        self.tracker = None
        self.update(None)
        self.stop_calc_time = None

    def select_followups(self, t, states, **kwargs):
        """Strategy for followups"""
        raise NotImplementedError("implement this")

    def stop_condition(self, t, radar, meta, tx_enu, rx_enu, tx_index, rx_index):
        if self.stop_calc_time is not None:
            return t > self.stop_calc_time
        else:
            return False

    def update(self, space_object, **kwargs):
        self.tracking_object = space_object

        if self.tracking_object is not None:
            dt = (self.tracking_object.epoch - self.epoch).to_value("sec")

            self.states_t = np.arange(dt, dt + self.max_predict_time, self.timeslice)

            if self.logger is not None:
                self.logger.info(
                    f"StareAndChase:update:propagating {len(self.states_t)} steps"
                )
            if self.profiler is not None:
                self.profiler.start("StareAndChase:update:propagating")

            self.states = self.tracking_object.get_state(self.states_t - dt)

            if self.profiler is not None:
                self.profiler.stop("StareAndChase:update:propagating")
            if self.logger is not None:
                self.logger.info(f"StareAndChase:update:propagating complete")

            self.passes = self.radar.find_passes(
                self.states_t, self.states, cache_data=False
            )

        self.scanner = sorts.controller.Scanner(
            self.radar,
            self.scan,
            t_slice=self.timeslice,
            t=np.arange(0, self.end_time, self.timeslice),
            profiler=self.profiler,
            logger=self.logger,
        )
        self.controllers = [self.scanner]
        if self.tracking_object is not None and len(self.passes[0][0]) > 0:

            t_track, states_track = self.select_followups(
                self.states_t[self.passes[0][0][0].inds],
                self.states[:, self.passes[0][0][0].inds],
                **kwargs,
            )

            self.tracker = sorts.controller.Tracker(
                radar=self.radar,
                t=t_track,
                t0=0.0,
                t_slice=self.timeslice,
                ecefs=states_track[:3, :],
                return_copy=True,
            )

            self.tracker.meta["target"] = self.tracking_object.oid

            scan_t_keep = np.full(self.scanner.t.shape, True, dtype=np.bool)
            for ti in range(len(self.scanner.t)):
                scan_t_keep[ti] = np.all(
                    np.abs(self.scanner.t[ti] - self.tracker.t) > self.timeslice
                )

            # remove all scanning during tracking
            self.scanner.t = self.scanner.t[scan_t_keep]

            self.controllers.append(self.tracker)

    def update_tracker(self, space_object, start_time, **kwargs):
        self.tracking_object = space_object
        dt = (self.tracking_object.epoch - self.epoch).to_value("sec")

        self.states = self.tracking_object.get_state(self.states_t - dt)

        t_track, states_track = self.select_followups(
            self.states_t[self.passes[0][0][0].inds],
            self.states[:, self.passes[0][0][0].inds],
            **kwargs,
        )

        inds = t_track >= start_time
        self.tracker.t = self.tracker.t[self.tracker.t < start_time]

        self.tracker = sorts.controller.Tracker(
            radar=self.radar,
            t=t_track[inds],
            t0=0.0,
            t_slice=self.timeslice,
            ecefs=states_track[:3, inds],
            return_copy=True,
        )

        self.tracker.meta["target"] = self.tracking_object.oid

        self.controllers.append(self.tracker)

    def get_controllers(self):
        return self.controllers

    def generate_schedule(self, t, generator):
        header = ["time", "TX-az", "TX-el", "controller", "target"]

        data = []
        for ind, (radar, meta) in enumerate(generator):
            row = [
                t[ind],
                radar.tx[0].beam.azimuth,
                radar.tx[0].beam.elevation,
                meta["controller_type"].__name__,
                meta.get("target", ""),
            ]
            data.append(row)

        return data, header
