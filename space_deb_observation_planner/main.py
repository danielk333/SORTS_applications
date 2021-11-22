#!/usr/bin/env python

'''
SST planner CLI
=================

'''
import pathlib
import argparse
import sys
import subprocess
import shutil
import pickle
import multiprocessing as mp
import importlib.util
import time
import logging

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time, TimeDelta

import sorts
import pyorb

import ndm_plotting
import sst_simulation
import ndm_wrapper
import configuration


def run_passes_planner():
    pass


def run_object_planner():
    pass


def run_fragmentation_planner():
    if not ndm_wrapper.check_setup():
        ndm_wrapper.setup()


def run_orbit_planner():
    pass


def get_base_object(config):
    


    if line1 is not None and line2 is not None:
        tles = [
            (
                line1,
                line2,
            ),
        ]
        logging.info('Using TLEs')
    else:
        tles = None
        state = np.array([
                config.getfloat('orbit', 'x'),
                config.getfloat('orbit', 'y'),
                config.getfloat('orbit', 'z'),
                config.getfloat('orbit', 'vx'),
                config.getfloat('orbit', 'vy'),
                config.getfloat('orbit', 'vz'),
            ], 
            dtype=np.float64,
        )
        coords = config.get('orbit', 'coordinate-system')
        epoch = Time(config.getfloat('orbit', 'epoch'), format='mjd')

    parameters = dict(
        d = config.getfloat('orbit', 'd'),
        A = config.getfloat('orbit', 'a'),
        m = config.getfloat('orbit', 'm'),
    )
    parameters = {key: val for key, val in parameters.items() if val is not None}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='\
        Plan SSA observations of a single object, \
        fragmentation event or of an entire orbit')
    parser.add_argument('radar', type=str, help='The observing radar system')
    parser.add_argument('config', type=str, help='Config file for the planning')
    parser.add_argument('output', type=str, help='Path to output results')
    parser.add_argument('propagator', type=str, help='Propagator to use')
    parser.add_argument(
        '--target',
        default=['object'],
        choices=[
            'passes',
            'object', 
            'fragmentation', 
            'orbit',
        ],
        nargs='+',
        help='Type of target for observation',
    )

    args = parser.parse_args()

    radar = getattr(sorts.radars, args.radar)

    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    config = configuration.get_config(pathlib.Path(args.config))

    cores = config.getint('general', 'cores')
    if cores is None or cores < 1:
        cores = 1

    custom_scheduler_file = config.get('general', 'scheduler-file')

    if custom_scheduler_file is not None:
        spec = importlib.util.spec_from_file_location(
            "custom_scheduler_module",
            str(pathlib.Path(custom_scheduler_file).resolve()),
        )
        custom_scheduler_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_scheduler_module)
        custom_scheduler = custom_scheduler_module.scheduler
        logging.info('Custom scheduler loaded, adding observation prediction')
        logging.info(custom_scheduler)
    else:
        custom_scheduler = None
        logging.info('No custom scheduler, using optimal SNR tracking')


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
    logging.info(f'Using propagator: {args.propagator}')

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
            pop.propagator_options['settings']['out_frame'] = 'ITRS'
        space_object = pop.get_object(0)
        space_object.parameters.update(parameters)

    # pop0 = sorts.population.tle_catalog(
    #     tles,
    #     sgp4_propagation = True, 
    # )
    # pop0.propagator_options['settings']['out_frame']='ITRS'
    # obj0 = pop0.get_object(0)
    
    # logging.info(space_object.propagator.settings)
    # logging.info(obj0.propagator.settings)

    # logging.info(space_object.get_state([0]))
    # logging.info(obj0.get_state([0]))
    # logging.info(obj0.get_state([0]) - space_object.get_state([0]))

    logging.info(space_object)

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

        fdt = TimeDelta(fragmentation_time*3600.0, format='sec')
        fragmentation_epoch = space_object.epoch + fdt

        nbm_folder = output / 'nbm_results'
        nbm_folder.mkdir(parents=True, exist_ok=True)
        nbm_results = (MODEL_DIR / 'bin' / 'outputs').glob('*.txt')
        for file in nbm_results:
            logging.info(f'Moving nasa-breakup-model result to output {nbm_folder / file.name}')
            shutil.copy(str(file), str(nbm_folder / file.name))

        ndm_plotting.plot(MODEL_DIR / 'bin' / 'outputs', nbm_folder / 'plots')
    else:
        nbm_param_file = None

    obs_epoch = config.getfloat('general', 'epoch', fallback=None)
    if obs_epoch is None:
        obs_epoch = space_object.epoch
    else:
        obs_epoch = Time(obs_epoch, format='mjd')

    logging.info(f'Planning epoch = {obs_epoch}')
    t_start = config.getfloat('general', 't_start')*3600.0
    t_end = config.getfloat('general', 't_end')*3600.0
    t_step = config.getfloat('general', 't_step')

    logging.info(f'Planning time = t0 + {t_start/3600.0} h -> t0 + {t_end/3600.0} h')

    profiler = sorts.profiling.Profiler()
    logger = sorts.profiling.get_logger()

    t = np.arange(t_start, t_end, t_step)
    logging.info(f'T=[{t[0]} s -> @ {len(t)} x {t_step} s steps -> {t[-1]} s]')
    states = space_object.get_state(t)

    scheduler = TrackingScheduler(
        radar = radar, 
        profiler = profiler, 
        logger = logger,
    )
    scheduler.set_tracker(t, states)
    interpolator = sorts.interpolation.Legendre8(states, t)

    data = scheduler.observe_passes(
        scheduler.passes, 
        space_object = space_object, 
        epoch = obs_epoch, 
        interpolator = interpolator,
        snr_limit = False,
    )

    out_data_txt = ''

    for txi in range(len(radar.tx)):
        for rxi in range(len(radar.rx)):
            for pi, ps in enumerate(scheduler.passes[txi][rxi]):
                ps_start = obs_epoch + TimeDelta(ps.start(), format='sec')
                ps_end = obs_epoch + TimeDelta(ps.end(), format='sec')
                out_data_txt += f'TX={txi} | RX={rxi} | Pass {pi} |'
                out_data_txt += f'Start: {ps_start.iso} -> End: {ps_end.iso}\n'

    if 'orbit' in args.target:
        __out_frame = space_object.out_frame
        space_object.out_frame = 'TEME'
        orbit_state = space_object.get_state([0.0])
        space_object.out_frame = __out_frame

        odatas = process_orbit(
            orbit_state,
            space_object.parameters,
            space_object.epoch,
            t,
            propagator,
            propagator_options,
            scheduler,
            obs_epoch,
            config.getint('orbit', 'samples', fallback=300),
        )
        with open(output / 'orbit_data.pickle', 'wb') as fh:
            pickle.dump(odatas, fh)

            fig_orb = plt.figure()
            axes = []
            for ind in range(1, len(radar.rx)+1):
                axes.append(fig_orb.add_subplot(2, len(radar.rx), ind))

                sn = []
                t = []
                for oid in range(len(odatas)):
                    for ind, d in enumerate(odatas[oid][0][ind-1]):
                        snm = np.argmax(d['snr'])
                        sn.append(d['snr'][snm])
                        t.append(d['t'][snm])
                t = np.array(t)
                sn = np.array(t)

                t_sort = np.argsort(t)
                t = t[t_sort]
                sn = sn[t_sort]

                axes[ind-1].plot(t/3600.0, 10*np.log10(sn), '.b')
                axes[ind-1].set_ylabel('Orbit sample SNR [dB]')
                axes[ind-1].set_xlabel('Time past epoch [h]')

            fig_orb.savefig(output / 'orbit_sampling_snr.png')


    if 'fragmentation' in args.target:
        fragment_scheduler = TrackingScheduler(
            radar = radar, 
            profiler = profiler, 
            logger = logger,
        )

        propagator_options['settings']['in_frame'] = 'TEME'
        fragment_data = []
        fragment_states = {}

        pbar = tqdm(total=len(cloud_data), desc='Processing fragments')

        if cores > 0:
            pool = mp.Pool(cores)
            reses = []

            for fid, fragment in enumerate(cloud_data):
                space_fragment = sorts.SpaceObject(
                    propagator,
                    propagator_options = propagator_options,
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
                        fragment_scheduler,
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
                    propagator_options = propagator_options,
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
                    fragment_scheduler,
                    obs_epoch,
                )

                fragment_data.append(fdata)
                fragment_states[f'{fid}'] = states_fragments
            
        pbar.close()
        with open(output / 'fragment_data.pickle', 'wb') as fh:
            pickle.dump(fragment_data, fh)
        np.savez(output / 'fragment_states.npz', **fragment_states)

        fragment_pass_data = []
        for txi in range(len(radar.tx)):
            fragment_pass_data.append([])
            for rxi in range(len(radar.rx)):
                fragment_pass_data[txi].append(dict(
                    peak_snr = [],
                    peak_time = [],
                ))
                for fid, fdata in enumerate(fragment_data):
                    for fps in fdata[txi][rxi]:
                        max_sn_ind = np.argmax(fps['snr'])
                        fragment_pass_data[txi][rxi]['peak_snr'].append(
                            fps['snr'][max_sn_ind]
                        )
                        fragment_pass_data[txi][rxi]['peak_time'].append(
                            fps['t'][max_sn_ind]
                        )

        with open(output / 'fragment_pass_data.pickle', 'wb') as fh:
            pickle.dump(fragment_pass_data, fh)

        # plot results
        fig_frag = plt.figure(figsize=(15, 15))
        ax = fig_frag.add_subplot(111, projection='3d')
        sorts.plotting.grid_earth(ax)
        for fid, states_fragments in fragment_states.items():
            ax.plot(states_fragments[0, :], states_fragments[1, :], states_fragments[2, :], 'b', alpha=0.2)
            ax.plot(states_fragments[0, -1], states_fragments[1, -1], states_fragments[2, -1], '.r', alpha=1)
        sorts.plotting.set_axes_equal(ax)

        fig_frag.savefig(output / 'fragment_orbits.png')
        plt.close(fig_frag)

        if config.getboolean('fragmentation', 'animate', fallback=False):
            output_anim = output / 'animation'
            output_anim.mkdir(exist_ok=True)

            fig_frag = plt.figure(figsize=(15, 15))
            ax = fig_frag.add_subplot(111, projection='3d')
            
            max_offset = 10
            step_size = 100
            for tid in tqdm(range(0, len(t), step_size)):
                if tid < max_offset:
                    offset = tid
                else:
                    offset = offset
                ax.clear()
                sorts.plotting.grid_earth(ax)
                for fid, states_fragments in fragment_states.items():
                    ax.plot(
                        states_fragments[0, (tid - offset):tid], 
                        states_fragments[1, (tid - offset):tid], 
                        states_fragments[2, (tid - offset):tid], 
                        'b', alpha=0.2,
                    )
                    ax.plot(
                        states_fragments[0, tid], 
                        states_fragments[1, tid], 
                        states_fragments[2, tid], 
                        '.r', alpha=1,
                    )
                if tid == 0:
                    sorts.plotting.set_axes_equal(ax)
                fig_frag.savefig(output_anim / f'frag_anim{tid}.png')
            
            plt.close(fig_frag)

            try:
                conv_path = pathlib.Path(".") / "animation" / "*.png"
                subprocess.check_call(
                    f'convert -delay 20 -loop 0 {str(conv_path)} animation.gif', 
                    cwd=str(output),
                )
            except subprocess.CalledProcessError as e:
                logging.info(e)
                logging.info('Could not create gif from animation frames... probably ImageMagick is missing')

        fig_frags, axes = plt.subplots(3, len(radar.rx))
        axes = []
        for ind in range(0, len(radar.rx)):

            sn = np.array(fragment_pass_data[0][ind-1]['peak_snr'])
            t = np.array(fragment_pass_data[0][ind-1]['peak_time'])

            ax = axes[0][ind]
            ax.hist(10*np.log10(sn[sn > 1]))
            ax.set_xlabel('Max SNR [dB]')
            ax.set_ylabel('Passes')

            ax = axes[1][ind]
            ax.hist(t[sn > 1]/3600.0)
            ax.set_xlabel('Time past epoch [h]')
            ax.set_ylabel('Passes')

            ax = axes[2][ind]
            ax.plot(t[sn > 1]/3600.0, 10*np.log10(sn[sn > 1]), '.b')
            ax.set_xlabel('Time past epoch [h]')
            ax.set_ylabel('Max SNR [dB]')

        fig_frags.savefig(output / 'fragments_snr_distributions.png')

    # plot results
    fig1 = plt.figure(figsize=(15, 15))
    fig2 = plt.figure(figsize=(15, 15))
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
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        
    for rxi, ax in enumerate(sn_axes):
        ax.set_xlabel('Time during pass [d]')
        ax.set_ylabel('SNR [dB/h]')
        ax.set_title(f'Receiver station {rxi}')

    r_axes[0].legend()
    for rxi, ax in enumerate(r_axes):
        ax.set_xlabel('Time during pass [d]')
        ax.set_ylabel('Range [km]')
        ax.set_title(f'Receiver station {rxi}')

    # plot results
    fig3 = plt.figure(figsize=(15, 15))
    ax = fig3.add_subplot(111, projection='3d')
    ax.plot(states[0, :]*1e-3, states[1, :]*1e-3, states[2, :]*1e-3, 'b')
    for ctrl in scheduler.controllers:
        for ti in range(len(ctrl.t)):
            ax.plot(
                [radar.tx[0].ecef[0]*1e-3, ctrl.ecefs[0, ti]*1e-3], 
                [radar.tx[0].ecef[1]*1e-3, ctrl.ecefs[1, ti]*1e-3], 
                [radar.tx[0].ecef[2]*1e-3, ctrl.ecefs[2, ti]*1e-3], 
                'g',
            )

    ax.set_xlabel('Time since epoch [d]')
    ax.set_ylabel('Distance from Earth [km]')

    fig1.savefig(output / 'kvec_snr.png')
    fig2.savefig(output / 'range.png')
    fig3.savefig(output / '3d_obs.png')
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

    with open(output / 'cmd.txt', 'w') as fh:
        fh.write(" ".join(sys.argv))

    with open(output / 'data.txt', 'w') as fh:
        fh.write(out_data_txt)

    with open(output / 'passes.pickle', 'wb') as fh:
        pickle.dump(scheduler.passes, fh)
