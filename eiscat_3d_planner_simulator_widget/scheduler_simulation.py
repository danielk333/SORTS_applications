#!/usr/bin/env python

'''
Observing a set of passes
================================

'''
import pathlib
import pickle
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

import sorts

p = sorts.profiling.Profiler()

def get_master_population(path):

    pop = sorts.population.master_catalog(
        path,
        propagator = sorts.propagator.SGP4,
        propagator_options = dict(
            settings = dict(
                in_frame='TEME',
                out_frame='ITRF',
            ),
        ),
    )
    pop = sorts.population.master_catalog_factor(pop, treshhold = 0.1, copy=False, seed=87346)

    return pop


def get_widget_limits(pop, END_T):
    lims = {
        'd_min': [1, np.max(pop['d']), 0.1],
        'L': [50, 200, 0.1],
        'N': [4, 100, 1],
        'dwell': [0.05, 2.0, 0.05],
        'h': [80, 400, 10],
        'h_min': [60, 400, 10],
        'beams': [1, 20, 1],
        't': [0, END_T-10.0, 0.1], #minutes
        't_span': [0.05, 60.0, 0.05], #sec
        'view_range': [300, 3000, 10],
    }
    return lims


class E3DScheduler(sorts.scheduler.PointingSchedule, sorts.scheduler.ObservedParameters):

    def __init__(self, radar, population, passes, states, dt, END_T, profiler=None, logger=None, **kwargs):
        super().__init__(
            radar=radar, 
            logger=logger, 
            profiler=profiler,
        )
        self.END_T = END_T
        self.dt = dt

        self.population = population
        self.t = np.arange(0, self.END_T, self.dt)

        self.scanner = None
        self.controllers = None
        self.passes = passes
        self.states = states
        self.inds = []

    def update(self, d_min, L, N, dwell, h, h_min, beams, track_ind):
        scan = sorts.scans.Plane(x_offset=0, altitude=h, x_size=L, y_size=L, x_num=N, y_num=N, dwell=dwell)
        self.scanner = sorts.controller.Scanner(self.radar, scan, t=np.arange(0, self.END_T, dwell), r=np.linspace(h_min, h, num=beams), as_altitude=True)

        self.controllers = [self.scanner]

        tracker_id = 1
        if track_ind is not None:
            for pi, ps in enumerate(self.passes[track_ind][0][0]):
                if ps is None:
                    continue
                tracker = sorts.controller.Tracker(radar=self.radar, t=self.t[ps.inds], ecefs=self.states[track_ind][:3,ps.inds], dwell=0.2)
                tracker.meta['id'] = tracker_id
                tracker.meta['target'] = track_ind
                tracker.meta['pass'] = pi
                self.controllers.append(tracker)

    def get_controllers(self):
        return self.controllers


def get_scheduler(radar, pop, dt, END_T):
    e3d_sched = E3DScheduler(radar, pop, {}, {}, dt, END_T)

    lims = get_widget_limits(pop, END_T)

    inds = np.squeeze(np.argwhere(e3d_sched.population.data['d'] > lims['d_min'][0])).tolist()
    e3d_sched.inds = inds

    cache_p = pathlib.Path('./passes.sim')
    cache_s = pathlib.Path('./states.sim')

    if not cache_p.is_file():
        passes = {}
        states = {}
        for ind in inds:
            obj = e3d_sched.population.get_object(ind)
            states[ind] = obj.get_state(e3d_sched.t)
            #set cache_data = True to save the data in local coordinates 
            #for each pass inside the Pass instance, setting to false saves RAM
            passes[ind] = e3d_sched.radar.find_passes(e3d_sched.t, states[ind], cache_data = False)

        with open(cache_p, 'wb') as h:
            pickle.dump(passes, h)
        with open(cache_s, 'wb') as h:
            pickle.dump(states, h)
    else:
        with open(cache_p, 'rb') as h:
            passes = pickle.load(h)
        with open(cache_s, 'rb') as h:
            states = pickle.load(h)

    e3d_sched.passes = passes
    e3d_sched.states = states

    return e3d_sched

alpha = 0.5

TRACK_DATA = None
TRACK_INDEX = None

def set_track_index(id_):
    global TRACK_INDEX
    TRACK_INDEX = id_

def recalc_track_data(e3d_sched):
    track_ind = TRACK_INDEX
    global TRACK_DATA

    data = e3d_sched.observe_passes(
        e3d_sched.passes[track_ind], 
        space_object = e3d_sched.population.get_object(track_ind), 
        snr_limit = False,
    )
    
    TRACK_DATA = data


def calc_observation(e3d_sched, ax, ax2, t, t_span, view_range, plot_rx, d_min, L, N, dwell, h, h_min, beams, include_objects, track_ind, track_snr):

    if track_ind == 'None':
        track_ind = None

    set_track_index(track_ind)

    e3d_sched.update(d_min, L*1e3, N, dwell, h*1e3, h_min*1e3, beams, track_ind)

    ax.clear()
    sorts.plotting.schedule_pointing(e3d_sched, t0=t, t1=t+t_span, ax=ax, plot_rx=plot_rx, view_range=view_range*1e3)

    if include_objects:
        inds = np.squeeze(np.argwhere(e3d_sched.population.data['d'] > d_min)).tolist()
        e3d_sched.inds = inds

        if isinstance(inds, int):
            inds = [inds]
        for ind in inds:
            for ps in e3d_sched.passes[ind][0][0]:
                if ps is None:
                    continue
                if ps.start() <= t + t_span and ps.end() >= t:
                    t_ind = np.argmin(np.abs(e3d_sched.t[ps.inds] - t))
                    ax.plot(e3d_sched.states[ind][0,ps.inds[t_ind]], e3d_sched.states[ind][1,ps.inds[t_ind]], e3d_sched.states[ind][2,ps.inds[t_ind]], 'ob', alpha=alpha)
                    ax.text(e3d_sched.states[ind][0,ps.inds[t_ind]], e3d_sched.states[ind][1,ps.inds[t_ind]], e3d_sched.states[ind][2,ps.inds[t_ind]], f'{ind}')
                    ax.plot(e3d_sched.states[ind][0,ps.inds], e3d_sched.states[ind][1,ps.inds], e3d_sched.states[ind][2,ps.inds], '-b', alpha=alpha)

    if track_snr:
        if TRACK_DATA is None:
            recalc_track_data(e3d_sched)

        data = TRACK_DATA

        ax2.set_visible(True)
        ax2.clear()

        for dat, ps in zip(data[0][0],e3d_sched.passes[track_ind][0][0]):
            if dat is None:
                continue
            st_ = np.argsort(dat['t'])
            track_inds = []
            for ti_ in st_:
                if dat['metas'][ti_]['controller_type'] == sorts.controller.Tracker:
                    track_inds.append(ti_)
            track_inds = np.array(track_inds)

            ax2.plot(dat['t'][st_] + ps.start(), 10*np.log10(dat['snr'][st_]), '-b')
            ax2.plot(dat['t'][track_inds] + ps.start(), 10*np.log10(dat['snr'][track_inds]), '.r')
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel('SNR [dB]')

        ax2.set_ylim([0, None])
        if ps.start() <= t + t_span and ps.end() >= t:
            ax2.axvline(t, color='r')
            ax2.axvline(t + t_span, color='r')
        
    else:
        ax2.set_visible(False)
        

    return ax, ax2
