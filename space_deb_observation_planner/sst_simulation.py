import pathlib
import multiprocessing as mp
import time
import copy
import logging

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time, TimeDelta

import sorts

logger = logging.getLogger('sst-cli.sst_simulation')


class TrackingScheduler(
            sorts.scheduler.StaticList, 
            sorts.scheduler.ObservedParameters,
        ):

    def __init__(self, radar, dwell=0.1, tracklet_point_spacing=1.0, profiler=None, logger=None, **kwargs):
        self.dwell = dwell
        self.tracklet_point_spacing = tracklet_point_spacing
        super().__init__(
            radar=radar, 
            controllers=None,
            logger=logger, 
            profiler=profiler,
            **kwargs
        )

    def set_tracker(self, t, states, interpolator=None):
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

                if interpolator is None:
                    inds = np.logical_and(t >= t_min, t <= t_max)
                    tracker = sorts.controller.Tracker(
                        radar = self.radar, 
                        t = t[inds], 
                        ecefs = states[:3, inds],
                        dwell = self.dwell,
                    )
                else:
                    t_track = np.arange(t_min, t_max, self.tracklet_point_spacing)
                    track_states = interpolator.get_state(t_track)
                    tracker = sorts.controller.Tracker(
                        radar = self.radar, 
                        t = t_track, 
                        ecefs = track_states[:3, :],
                        dwell = self.dwell,
                    )
                self.controllers.append(tracker)


def _process_orbit_job(anom, space_orbit, radar, t, obs_epoch, custom_scheduler=None):
    if custom_scheduler is None:
        scheduler = TrackingScheduler(
            radar = radar, 
        )
    else:
        scheduler = custom_scheduler

    space_orbit.orbit.anom = anom

    states_orb = space_orbit.get_state(t)
    interpolator = sorts.interpolation.Legendre8(states_orb, t)
    scheduler.set_tracker(t, states_orb)
    odata = scheduler.observe_passes(
        scheduler.passes, 
        space_object = space_orbit, 
        epoch = obs_epoch, 
        interpolator = interpolator,
        snr_limit = False,
    )
    return odata


def process_orbit(
            state,
            parameters,
            epoch,
            t,
            propagator,
            propagator_options,
            radar,
            profiler,
            obs_epoch,
            num,
            cores=0,
            custom_scheduler_getter=None,
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

    if cores > 1:
        pool = mp.Pool(cores)
        reses = []

        pbar = tqdm(total=num, desc='Orbit mean anomaly sampling')
        for ai, anom in enumerate(np.linspace(0, 360.0, num=num)):
            if custom_scheduler_getter is not None:
                _kwargs = dict(custom_scheduler=custom_scheduler_getter())
            else:
                _kwargs = {}

            reses.append(pool.apply_async(
                _process_orbit_job, 
                args=(
                    anom, 
                    space_orbit.copy(),
                    radar.copy(),
                    t,
                    obs_epoch,
                ), 
                kwds=_kwargs,
            ))
        pool_status = np.full((num, ), False)
        while not np.all(pool_status):
            for fid, res in enumerate(reses):
                if pool_status[fid]:
                    continue
                time.sleep(0.001)
                _ready = res.ready()
                if not pool_status[fid] and _ready:
                    pool_status[fid] = _ready
                    pbar.update()

        for fid, res in enumerate(reses):
            odata = res.get()
            odatas.append(odata)

        pool.close()
        pool.join()
        pbar.close()

    else:
        if custom_scheduler_getter is not None:
            scheduler = custom_scheduler_getter()
        else:
            scheduler = TrackingScheduler(
                radar = radar, 
                profiler = profiler, 
                logger = logger,
            )
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
            radar,
            tracklet_point_spacing,
            obs_epoch,
        ):

    scheduler = TrackingScheduler(
        radar = radar, 
        tracklet_point_spacing = tracklet_point_spacing,
    )

    states_fragments = space_object.get_state(t)
    interpolator = sorts.interpolation.Legendre8(states_fragments, t)
    scheduler.set_tracker(t, states_fragments, interpolator=interpolator)
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
    # TODO: This is ugly and wrong as shit, FIX IT
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
        'B': B,
        'A': A,
        'm': m,
        'd': r*2.0,
        'C_D': C_D,
        'C_R': 1.0,
    })
    space_object.parameters.update(parameters)

    return space_object


def observe_nbm_fragments(
            fragmentation_epoch, 
            cloud_data, 
            radar,
            tracklet_point_spacing, 
            obs_epoch,
            t,
            propagator, 
            propagator_options, 
            cores,
        ):

    propagator_options_local = copy.deepcopy(propagator_options)
    propagator_options_local['settings']['in_frame'] = 'TEME'
    fragment_data = []
    fragment_states = {}

    pbar = tqdm(total=len(cloud_data), desc='Processing fragments')

    if cores > 1:
        pool = mp.Pool(cores)
        reses = []

        for fid, fragment in enumerate(cloud_data):
            space_fragment = sorts.SpaceObject(
                propagator,
                propagator_options = propagator_options_local,
                state = fragment[4:10],
                epoch = fragmentation_epoch,
                parameters = dict(
                    m = fragment[11],
                    d = fragment[12],
                ),
            )
            reses.append(pool.apply_async(
                process_object, 
                args=(
                    space_fragment,
                    t,
                    radar,
                    tracklet_point_spacing,
                    obs_epoch,
                ), 
            ))
        pool_status = np.full((len(cloud_data), ), False)
        while not np.all(pool_status):
            for fid, res in enumerate(reses):
                if pool_status[fid]:
                    continue
                time.sleep(0.01)
                _ready = res.ready()
                if not pool_status[fid] and _ready:
                    pool_status[fid] = _ready
                    pbar.update()

        for fid, res in enumerate(reses):
            fdata, states_fragments = res.get()
            fragment_data.append(fdata)
            fragment_states[f'{fid}'] = states_fragments

        pool.close()
        pool.join()

    else:
        for fid, fragment in enumerate(cloud_data):
            pbar.update()
            space_fragment = sorts.SpaceObject(
                propagator,
                propagator_options = propagator_options_local,
                state = fragment[4:10],
                epoch = fragmentation_epoch,
                parameters = dict(
                    m = fragment[11],
                    d = fragment[12],
                ),
            )
            fdata, states_fragments = process_object(
                space_fragment,
                t,
                radar,
                tracklet_point_spacing,
                obs_epoch,
            )

            fragment_data.append(fdata)
            fragment_states[f'{fid}'] = states_fragments
    
    fragment_pass_data = []
    for txi in range(len(radar.tx)):
        fragment_pass_data.append([])
        for rxi in range(len(radar.rx)):

            # TODO: add more statistics here
            fragment_pass_data[txi].append(dict(
                peak_snr = [],
                peak_time = [],
            ))
            for fid, fdata in enumerate(fragment_data):
                for fps in fdata[txi][rxi]:
                    if fps is None:
                        continue
                    max_sn_ind = np.argmax(fps['snr'])
                    fragment_pass_data[txi][rxi]['peak_snr'].append(
                        fps['snr'][max_sn_ind]
                    )
                    fragment_pass_data[txi][rxi]['peak_time'].append(
                        fps['t'][max_sn_ind]
                    )

    return fragment_pass_data, fragment_data, fragment_states


def _ugly_test1():
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

    space_object = convert_tle_so_to_state_so(space_object_tle, samples=100, sample_time=60*90.0)
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
    _ugly_test1()
