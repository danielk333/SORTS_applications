import pathlib
import configparser
import argparse
import sys
import subprocess
import shutil
import pickle
import multiprocessing as mp
import importlib.util
import time
import copy

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time, TimeDelta

import sorts
import pyorb


class TrackingScheduler(
            sorts.scheduler.StaticList, 
            sorts.scheduler.ObservedParameters,
        ):

    def __init__(self, radar, dwell=0.1, profiler=None, logger=None, **kwargs):
        self.dwell = dwell
        super().__init__(
            radar=radar, 
            controllers=None,
            logger=logger, 
            profiler=profiler,
            **kwargs
        )

    def set_tracker(self, t, states):
        self.passes = self.radar.find_passes(t, states)
        passes = sorts.passes.group_passes(self.passes)

        self.controllers = []
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
                    radar = self.radar, 
                    t = t[inds], 
                    ecefs = states[:3, inds],
                    dwell = self.dwell,
                )
                self.controllers.append(tracker)


def process_orbit(
            state,
            parameters,
            epoch,
            t,
            propagator,
            propagator_options,
            scheduler,
            obs_epoch,
            num,
        ):
    space_orbit = sorts.SpaceObject(
        propagator,
        propagator_options = propagator_options,
        x = state[0],
        y = state[1],
        z = state[2],
        vx = state[3],
        vy = state[4],
        vz = state[5],
        epoch = epoch,
        parameters = parameters,
    )
    space_orbit.in_frame = 'TEME'
    space_orbit.out_frame = 'ITRS'
    space_orbit.orbit.type = 'mean'
    space_orbit.orbit.degrees = True

    odatas = []
    for anom in tqdm(np.linspace(0, 360.0, num=num)):
        space_orbit.orbit.anom = anom

        states_orb = space_orbit.get_state(t)
        interpolator = sorts.interpolation.Legendre8(states_orb, t)
        scheduler.set_tracker(t, states_orb)
        odatas += [scheduler.observe_passes(
            scheduler.passes, 
            space_object = space_orbit, 
            epoch = obs_epoch, 
            interpolator = interpolator,
            snr_limit = False,
        )]
    return odatas


def process_object(
            space_object,
            t,
            scheduler,
            obs_epoch,
        ):

    states_fragments = space_object.get_state(t)
    interpolator = sorts.interpolation.Legendre8(states_fragments, t)
    scheduler.set_tracker(t, states_fragments)
    fdata = scheduler.observe_passes(
        scheduler.passes, 
        space_object = space_object, 
        epoch = obs_epoch, 
        interpolator = interpolator,
        snr_limit = False,
    )
    return fdata, states_fragments


def convert_tle_so_to_state_so(
            tle_space_object,
            propagator = None,
            propagator_options = None,
            samples = 100,
            sample_time = 3600*24.0,
        ):

    __out_frame = tle_space_object.out_frame
    tle_space_object.out_frame = 'TEME'
    if samples <= 1:
        t = np.array([0.0])
    else:
        t = np.linspace(0.0, sample_time, num=samples)
    state_TEME = tle_space_object.get_state(t)
    tle_space_object.out_frame = __out_frame

    if propagator is None:
        propagator = sorts.propagator.SGP4

    if propagator_options is None:
        propagator_options = {
            'settings': {
                'out_frame': 'ITRS',
            },
        }

    if 'settings' not in propagator_options:
        propagator_options['settings'] = {}

    propagator_options['settings']['in_frame'] = 'TEME'
    space_object = sorts.SpaceObject(
        propagator,
        propagator_options = propagator_options,
        propagator_args = {
            'state_sample_times': t,
        },
        epoch = copy.deepcopy(tle_space_object.epoch),
        state = state_TEME,
        oid = tle_space_object.oid,
        parameters = copy.deepcopy(tle_space_object.parameters),
    )

    return space_object


def get_space_object_from_tle(line1, line2, parameters):

    params = sorts.propagator.SGP4.get_TLE_parameters(line1, line2)
    bstar = params['bstar']
    epoch = Time(params['jdsatepoch'] + params['jdsatepochF'], format='jd', scale='utc')
    oid = params['satnum']

    space_object = sorts.SpaceObject(
        sorts.propagator.SGP4,
        propagator_options = dict(
            settings = dict(
                out_frame='ITRS',
                tle_input=True,
            ),
        ),
        state = [line1, line2],
        epoch = epoch,
        parameters = {},
        oid = oid,
    )

    bstar = bstar/(space_object.propagator.grav_model.radiusearthkm*1000.0)
    B = bstar*2.0/space_object.propagator.rho0
    if B < 1e-9:
        rho = 500.0
        C_D = 0.0
        r = 0.1
        A = np.pi*r**2
        m = rho*4.0/3.0*np.pi*r**3
    else:
        C_D = 2.3
        rho = 5.0
        r = (3.0*C_D)/(B*rho)
        A = np.pi*r**2
        m = rho*4.0/3.0*np.pi*r**3

    space_object.parameters.update({
        'A': A,
        'm': m,
        'd': r*2.0,
        'C_D': C_D,
        'C_R': 1.0,
    })
    space_object.parameters.update(parameters)

    return space_object


def ugly_test1():
    import configuration
    config = configuration.get_config(pathlib.Path('./example.conf'))

    line1 = config.get('orbit', 'line 1')
    line2 = config.get('orbit', 'line 2')

    parameters = dict(
        d = config.custom_getfloat('orbit', 'd'),
        A = config.custom_getfloat('orbit', 'a'),
        m = config.custom_getfloat('orbit', 'm'),
    )
    parameters = {key: val for key, val in parameters.items() if val is not None}

    space_object_tle = get_space_object_from_tle(line1, line2, parameters)

    print(space_object_tle)

    t = np.linspace(0, 3600*24.0, num=5000)

    states_tle = space_object_tle.get_state(t)

    space_object = convert_tle_so_to_state_so(space_object_tle, samples=1000, sample_time=60*90.0)
    space_object_bad = convert_tle_so_to_state_so(space_object_tle, samples=1)

    states_itrs = space_object.get_state(t)
    states_itrs_bad = space_object_bad.get_state(t)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states_tle[0, :], states_tle[1, :], states_tle[2, :], "-k")
    ax.plot(states_itrs[0, :], states_itrs[1, :], states_itrs[2, :], "-r")
    ax.plot(states_itrs_bad[0, :], states_itrs_bad[1, :], states_itrs_bad[2, :], "-b")
    ax.set_title('TLE versus State propagation /w SGP4')

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(t/3600.0, np.log10(np.sqrt(np.sum((states_itrs[:3, :] - states_tle[:3, :])**2, axis=0))))
    axes[0, 0].set_xlabel('Time past epoch [h]')
    axes[0, 0].set_ylabel('Mean element error [log10(m)]')
    axes[1, 0].plot(t/3600.0, np.sqrt(np.sum((states_itrs[:3, :] - states_tle[:3, :])**2, axis=0)))
    axes[1, 0].set_xlabel('Time past epoch [h]')
    axes[1, 0].set_ylabel('Mean element error [m]')
    axes[0, 1].plot(t/3600.0, np.log10(np.sqrt(np.sum((states_itrs_bad[:3, :] - states_tle[:3, :])**2, axis=0))))
    axes[0, 1].set_xlabel('Time past epoch [h]')
    axes[0, 1].set_ylabel('Mean element error [log10(m)]')
    axes[1, 1].plot(t/3600.0, np.sqrt(np.sum((states_itrs_bad[:3, :] - states_tle[:3, :])**2, axis=0)))
    axes[1, 1].set_xlabel('Time past epoch [h]')
    axes[1, 1].set_ylabel('Mean element error [m]')
    plt.show()


if __name__ == '__main__':
    ugly_test1()
