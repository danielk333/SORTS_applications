#!/usr/bin/env python

'''
E3D Demonstrator SST planner
================================

'''
import pathlib
import configparser
import argparse
import sys
import subprocess
import shutil

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time, TimeDelta

import sorts
import pyorb

import ndm_plotting

HERE = pathlib.Path(__file__).parent.resolve()
CACHE = HERE / '.cache'
MODEL_DIR = CACHE  / 'nasa-breakup-model'

def check_setup():
    bin_file = CACHE / 'nasa-breakup-model' / 'bin' / 'breakup'
    return bin_file.is_file()


def setup():
    '''Setup requrements for this CLI
    '''

    CACHE.mkdir(parents=False, exist_ok=True)

    get_cmd = ['git', 'clone', 'https://gitlab.obspm.fr/apetit/nasa-breakup-model.git']

    if not MODEL_DIR.is_dir():
        subprocess.check_call(get_cmd, cwd=str(MODEL_DIR))
    
    subprocess.check_call(['make'], cwd=str(MODEL_DIR / 'src'))



def run_nasa_breakup(nbm_param_file, nbm_param_type, space_object, fragmentation_time):
    old_inps = (MODEL_DIR / 'bin' / 'inputs').glob('nbm_param_[0-9]*.txt')
    for file in old_inps:
        print(f'Removing old input file {file}')
        file.unlink()

    shutil.copy(nbm_param_file, str(MODEL_DIR / 'bin' / 'inputs' / 'nbm_param_1.txt'))

    event_time = space_object.epoch + TimeDelta(fragmentation_time*3600.0, format='sec')

    state_itrs = space_object.get_state([fragmentation_time*3600.0])
    state_teme = sorts.frames.convert(
        event_time, 
        state_itrs, 
        in_frame='ITRS', 
        out_frame='TEME',
    )
    orbit_teme = pyorb.Orbit(
        cartesian=state_teme,
        M0 = pyorb.M_earth,
        m = space_object.parameters.get('m', 0.0),
        degrees=True,
        type='mean',
    )
    kepler_teme = orbit_teme.kepler.flatten()

    sched_header = ['ID_NORAD', 'J_DAY', 'YYYY', 'MM', 'DD', 'NAME', 'TYPE', 'SMA[m]', 'ECC', 'INC[deg]', 'RAAN[deg]', 'A_PER[deg]', 'MA[deg]', 'MASS[kg]', 'S']
    sched_data = [
        '00000001', 
        f'{event_time.jd:.6f}', 
        str(event_time.datetime.year), 
        str(event_time.datetime.month), 
        str(event_time.datetime.day),
        str(space_object.oid),
        nbm_param_type,
        f'{kepler_teme[0]:.8e}',
        f'{kepler_teme[1]:.8e}',
        f'{kepler_teme[2]:.8e}',
        f'{kepler_teme[4]:.8e}',
        f'{kepler_teme[3]:.8e}',
        f'{kepler_teme[5]:.8e}',
        str(space_object.parameters.get('m', 0.0)),
        str(1.0), #scale factor
    ]

    max_key_len = max([len(x) for x in sched_header])
    print('Fragmentation model with object:')
    for key, val in zip(sched_header, sched_data):
        print(f'{key:<{max_key_len}}: {val}')

    sched_file = MODEL_DIR / 'bin' / 'inputs' / 'schedule.txt'
    print(f'Writing schedule file: {sched_file}')
    with open(sched_file, 'w') as fh:
        fh.write(' '.join(sched_header) + '\n')
        fh.write(' '.join(sched_data))

    print('Running nasa-breakup-model...')
    subprocess.check_call(['./breakup'], cwd=str(MODEL_DIR / 'bin'))
    print('nasa-breakup-model done')
    
    data_file = MODEL_DIR / 'bin' / 'outputs' / 'cloud_cart.txt'

    cloud_data = np.genfromtxt(
        str(data_file),
        skip_header=1,
    )
    with open(data_file, 'r') as fh:
        header = fh.readline().split()

    return cloud_data, header

class TrackingScheduler(
        sorts.scheduler.StaticList, 
        sorts.scheduler.ObservedParameters,
    ):
    def __init__(self, radar, t, states, dwell=0.1, profiler=None, logger=None, **kwargs):
        self.dwell = dwell
        super().__init__(
            radar=radar, 
            controllers=None,
            logger=logger, 
            profiler=profiler,
            **kwargs
        )
        self.set_tracker(t, states)


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




if __name__=='__main__':

    if not check_setup():
        setup()

    parser = argparse.ArgumentParser(description='Plan SSA observations of a single object, fragmentation event or of an entire orbit')
    parser.add_argument('radar', type=str, help='The observing radar system')
    parser.add_argument('config', type=str, help='Config file for the planning')
    parser.add_argument('output', type=str, help='Path to output results')
    parser.add_argument('propagator', type=str, help='Propagator to use')
    parser.add_argument('--target', default=['object'], choices=['object', 'fragmentation', 'orbit'], nargs='+', help='Type of target for observation')

    args = parser.parse_args()

    radar = getattr(sorts.radars, args.radar)

    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    config = configparser.ConfigParser(interpolation=None)
    config.read([args.config])

    line1 = config.get('orbit', 'line 1', fallback=None)
    line2 = config.get('orbit', 'line 2', fallback=None)

    if line1 is not None and line2 is not None:
        tles = [
            (
                line1,
                line2,
             ),
        ]
        print('Using TLEs')
    else:
        tles = None
        state = np.array([
                config.getfloat('orbit', 'x', fallback=None),
                config.getfloat('orbit', 'y', fallback=None),
                config.getfloat('orbit', 'z', fallback=None),
                config.getfloat('orbit', 'vx', fallback=None),
                config.getfloat('orbit', 'vy', fallback=None),
                config.getfloat('orbit', 'vz', fallback=None),
            ], 
            dtype=np.float64,
        )
        coords = config.get('orbit', 'coordinate-system')
        epoch = Time(config.getfloat('orbit', 'epoch'), format='mjd')

    parameters = dict(
        d = config.getfloat('orbit', 'd', fallback=None),
        A = config.getfloat('orbit', 'a', fallback=None),
        m = config.getfloat('orbit', 'm', fallback=None),
    )
    parameters = {key:val for key,val in parameters.items() if val is not None}

    if args.propagator == 'SGP4':
        if 'fragmentation' not in args.target and 'orbit' not in args.target:
            sgp4_propagation = True
            propagator = None
            propagator_options = {}
        else:
            sgp4_propagation = False
            propagator = sorts.propagator.SGP4
            propagator_options = dict(
                settings=dict(
                    in_frame='TEME',
                    out_frame='ITRS',
                )
            )
    elif args.propagator == 'Orekit':
        orekit_data = config.get('general', 'orekit-data', fallback=None)
        if orekit_data is None:
            orekit_data = CACHE / 'orekit-data-master.zip'
            if not orekit_data.is_file():
                sorts.propagator.Orekit.download_quickstart_data(orekit_data, verbose=True)

        else:
            orekit_data = pathlib.Path(orekit_data)
            if not orekit_data.is_file():
                raise ValueError('Given orekit file does not exist')

        propagator = sorts.propagator.Orekit
        propagator_options = dict(
            orekit_data = orekit_data, 
            settings=dict(
                in_frame='TEME',
                out_frame='ITRS',
                drag_force = True,
                radiation_pressure = False,
            )
        )
    else:
        raise ValueError('No recognized propagator given')
    print(f'Using propagator: {args.propagator}')

    if tles is None:
        propagator_options['settings']['in_frame'] = coords
        space_object = sorts.SpaceObject(
            propagator,
            propagator_options = propagator_options,
            state = state,
            epoch = epoch,
            parameters = parameters,
        )
    else:
        pop = sorts.population.tle_catalog(
            tles,
            sgp4_propagation = sgp4_propagation, 
            cartesian = True,
            propagator = propagator,
            propagator_options = propagator_options,
        )
        if sgp4_propagation:
            pop.propagator_options['settings']['out_frame']='ITRS'
        space_object = pop.get_object(0)
        space_object.parameters.update(parameters)

    # pop0 = sorts.population.tle_catalog(
    #     tles,
    #     sgp4_propagation = True, 
    # )
    # pop0.propagator_options['settings']['out_frame']='ITRS'
    # obj0 = pop0.get_object(0)
    
    # print(space_object.propagator.settings)
    # print(obj0.propagator.settings)

    # print(space_object.get_state([0]))
    # print(obj0.get_state([0]))
    # print(obj0.get_state([0]) - space_object.get_state([0]))

    print(space_object)

    if 'fragmentation' in args.target:
        nbm_param_defaults = {
            'sc': MODEL_DIR / 'bin' / 'inputs' / 'nbm_param_sc.txt',
            'rb': MODEL_DIR / 'bin' / 'inputs' / 'nbm_param_rb.txt',
        }

        nbm_param_file = config.get('fragmentation', 'nbm_param_file', fallback=None)
        nbm_param_default = config.get('fragmentation', 'nbm_param_default', fallback=None)
        nbm_param_type = config.get('fragmentation', 'nbm_param_type')

        if nbm_param_file is None:
            if nbm_param_default is None:
                raise ValueError('No nasa-breakup-model parameter file given')
            nbm_param_file = str(nbm_param_defaults[nbm_param_default])

        fragmentation_time = config.getfloat('fragmentation', 'fragmentation_time')

        cloud_data, cloud_header = run_nasa_breakup(
            nbm_param_file, 
            nbm_param_type, 
            space_object, 
            fragmentation_time,
        )

        nbm_folder = output / 'nbm_results'
        nbm_folder.mkdir(parents=True, exist_ok=True)
        nbm_results = (MODEL_DIR / 'bin' / 'outputs').glob('*.txt')
        for file in nbm_results:
            print(f'Moving nasa-breakup-model result to output {nbm_folder / file.name}')
            shutil.copy(str(file), str(nbm_folder / file.name))

        ndm_plotting.plot(MODEL_DIR / 'bin' / 'outputs', nbm_folder / 'plots')
    else:
        nbm_param_file = None

    obs_epoch = config.getfloat('general', 'epoch', fallback=None)
    if obs_epoch is None:
        obs_epoch = space_object.epoch
    else:
        obs_epoch = Time(obs_epoch, format='mjd')

    print(f'Planning epoch = {obs_epoch}')
    t_start = config.getfloat('general', 't_start')*3600.0
    t_end = config.getfloat('general', 't_end')*3600.0
    t_step = config.getfloat('general', 't_step')

    print(f'Planning time = t0 + {t_start/3600.0} h -> t0 + {t_end/3600.0} h')

    profiler = sorts.profiling.Profiler()
    logger = sorts.profiling.get_logger()

    t = np.arange(t_start, t_end, t_step)
    states = space_object.get_state(t)

    scheduler = TrackingScheduler(
        radar = radar, 
        t = t,
        states = states, 
        profiler = profiler, 
        logger = logger,
    )

    data = scheduler.observe_passes(
        scheduler.passes, 
        space_object=space_object, 
        epoch=obs_epoch, 
        snr_limit=False,
    )

    out_data_txt = ''

    for txi in range(len(radar.tx)):
        for rxi in range(len(radar.rx)):
            for pi, ps in enumerate(scheduler.passes[txi][rxi]):
                ps_start = obs_epoch + TimeDelta(ps.start(), format='sec')
                ps_end = obs_epoch + TimeDelta(ps.end(), format='sec')
                out_data_txt += f'TX={txi} | RX={rxi} | Pass {pi} | Start: {ps_start.iso} -> End: {ps_end.iso}\n'

    #plot results
    fig1 = plt.figure(figsize=(15,15))
    fig2 = plt.figure(figsize=(15,15))
    axes = []
    r_axes = []
    sn_axes = []
    for ind in range(1, len(radar.rx)+1):
        axes.append(fig1.add_subplot(2, len(radar.rx), ind))
        r_axes.append(fig1.add_subplot(2, len(radar.rx), len(radar.rx) + ind))
        sn_axes.append(fig2.add_subplot(1, len(radar.rx), ind))

    for tx_d in data:
        for rxi, rx_d in enumerate(tx_d):
            for dati, dat in enumerate(rx_d):
                axes[rxi].plot(dat['tx_k'][0,:], dat['tx_k'][1,:], label=f'Pass {dati}')
                sn_axes[rxi].plot((dat['t'] - np.min(dat['t']))/(3600.0*24), 10*np.log10(dat['snr']), label=f'Pass {dati}')
                r_axes[rxi].plot((dat['t'] - np.min(dat['t']))/(3600.0*24), (dat['range']*0.5)*1e-3, label=f'Pass {dati}')

    axes[0].legend()
    for rxi, ax in enumerate(axes):
        ax.set_xlabel('k_x [East]')
        ax.set_ylabel('k_y [North]')
        ax.set_title(f'Receiver station {rxi}')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        

    for rxi, ax in enumerate(sn_axes):
        ax.set_xlabel('Time during pass [d]')
        ax.set_ylabel('SNR [dB/h]')
        ax.set_title(f'Receiver station {rxi}')

    r_axes[0].legend()
    for rxi, ax in enumerate(r_axes):
        ax.set_xlabel('Time during pass [d]')
        ax.set_ylabel('Range [km]')
        ax.set_title(f'Receiver station {rxi}')

    #plot results
    fig3 = plt.figure(figsize=(15,15))
    ax = fig3.add_subplot(111, projection='3d')
    ax.plot(states[0,:]*1e-3, states[1,:]*1e-3, states[2,:]*1e-3, 'b')
    for ctrl in scheduler.controllers:
        for ti in range(len(ctrl.t)):
            ax.plot(
                [radar.tx[0].ecef[0]*1e-3, ctrl.ecefs[0,ti]*1e-3], 
                [radar.tx[0].ecef[1]*1e-3, ctrl.ecefs[1,ti]*1e-3], 
                [radar.tx[0].ecef[2]*1e-3, ctrl.ecefs[2,ti]*1e-3], 
                'g',
            )

    ax.set_xlabel('Time since epoch [d]')
    ax.set_ylabel('Distance from Earth [km]')

    fig1.savefig(output / 'kvec_snr.png')
    fig2.savefig(output / 'range.png')
    fig3.savefig(output / '3d_obs.png')

    with open(output / 'cmd.txt', 'w') as fh:
        fh.write(" ".join(sys.argv))

    with open(output / 'data.txt', 'w') as fh:
        fh.write(out_data_txt)
