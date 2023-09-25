#!/usr/bin/env python

'''
SST planner CLI
=================

'''
import pathlib
import argparse
import pickle
import shutil
import logging
import sys


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
from astropy.time import Time, TimeDelta

import sorts

import ndm_plotting
import sst_simulation
import ndm_wrapper
import configuration as cfg
import plotting

logger = logging.getLogger('sst-cli')
profiler = sorts.profiling.Profiler()


class delayed_setdefault(dict):
    def get_cache(self, key, default_method):
        if key not in self:
            self[key] = default_method()
        return self[key]


def run_passes_planner(args, config, cores, radar, output, CACHE, profiler=None):
    space_object, propagator, propagator_options = CACHE.get_cache(
        'space_object',
        lambda: get_base_object(config, args),
    )
    t_data = CACHE.get_cache(
        't_data',
        lambda: get_times(config, space_object),
    )
    obs_epoch, t, t_start, t_end, t_step = t_data
    states = CACHE.get_cache(
        'states',
        lambda: space_object.get_state(t),
    )
    dt = (obs_epoch - space_object.epoch).sec

    passes = CACHE.get_cache(
        'passes',
        lambda: radar.find_passes(t - dt, states, cache_data = True),
    )

    passes_output = output / 'passes'
    passes_output.mkdir(exist_ok=True)

    with open(passes_output / 'passes.pickle', 'wb') as fh:
        pickle.dump(passes, fh)

    out_data_txt = ''

    fig, axes = plt.subplots(len(radar.tx), len(radar.rx))
    if len(radar.tx) == 1 and len(radar.rx) == 1:
        axes = np.array([axes])
    axes.shape = (len(radar.tx), len(radar.rx))

    for txi in range(len(radar.tx)):
        for rxi in range(len(radar.rx)):

            sorts.plotting.local_passes(passes[txi][rxi], ax=axes[txi, rxi])
            axes[txi, rxi].set_title(f'TX={txi} | RX={rxi}')

            for pi, ps in enumerate(passes[txi][rxi]):
                ps_start = obs_epoch + TimeDelta(ps.start(), format='sec')
                ps_end = obs_epoch + TimeDelta(ps.end(), format='sec')
                out_data_txt += f'TX={txi} | RX={rxi} | Pass {pi} |'
                out_data_txt += f'Start: {ps_start.iso} -> End: {ps_end.iso}\n'
    
    fig.savefig(passes_output / 'local_passes.png')
    with open(passes_output / 'pass_times.txt', 'w') as fh:
        fh.write(out_data_txt)

    logger.info('run_passes_planner COMPLETE')


def run_object_planner(args, config, cores, radar, output, CACHE, profiler=None):
    space_object, propagator, propagator_options = CACHE.get_cache(
        'space_object',
        lambda: get_base_object(config, args),
    )
    t_data = CACHE.get_cache(
        't_data',
        lambda: get_times(config, space_object),
    )
    obs_epoch, t, t_start, t_end, t_step = t_data
    states = CACHE.get_cache(
        'states',
        lambda: space_object.get_state(t),
    )

    interpolator = sorts.interpolation.Legendre8(states, t)

    scheduler_modes = [None]
    custom_scheduler_getter = get_custom_scheduler_getter(config, output)
    if custom_scheduler_getter is not None:
        scheduler_modes.append(custom_scheduler_getter)

    for mode in scheduler_modes:
        if mode is not None:
            logger.info('Running custom scheduler scheduler')
            file_modifier = 'custom_scheduler_'
            scheduler = mode()
            passes = scheduler.radar.find_passes(t, states)
        else:
            logger.info('Running optimal tracking scheduler')
            file_modifier = ''

            scheduler = sst_simulation.TrackingScheduler(
                radar = radar, 
                profiler = profiler, 
                logger = logger,
            )
            scheduler.set_tracker(t, states, interpolator=interpolator)
            passes = scheduler.passes
        
        tracking_data = scheduler.observe_passes(
            passes, 
            space_object = space_object, 
            epoch = obs_epoch, 
            interpolator = interpolator,
            snr_limit = False,
        )

        object_output = output / 'object'
        object_output.mkdir(exist_ok=True)

        with open(object_output / f'{file_modifier}tracking_data.pickle', 'wb') as fh:
            pickle.dump(tracking_data, fh)

        figsize = (
            config.custom_getint('general', 'figsize-x'),
            config.custom_getint('general', 'figsize-y'),
        )
        figsize = tuple([x for x in figsize if x is not None])
        if len(figsize) < 2:
            figsize = None

        fig1, fig2 = plotting.observed_pass_data(
            scheduler.radar, 
            tracking_data, 
            time_unit='min',
            obs_epoch=obs_epoch, 
            figsize=figsize,
        )
        fig1.savefig(object_output / f'{file_modifier}tracking_kv_range.png')
        fig2.savefig(object_output / f'{file_modifier}tracking_snr.png')


def run_fragmentation_planner(args, config, cores, radar, output, CACHE, profiler=None):
    if not ndm_wrapper.check_setup():
        ndm_wrapper.setup()

    fragmentation_output = output / 'fragmentation'
    fragmentation_output.mkdir(parents=True, exist_ok=True)

    space_object, propagator, propagator_options = CACHE.get_cache(
        'space_object',
        lambda: get_base_object(config, args),
    )
    t_data = CACHE.get_cache(
        't_data',
        lambda: get_times(config, space_object),
    )
    obs_epoch, t, t_start, t_end, t_step = t_data

    nbm_param_defaults = {
        'sc': cfg.MODEL_DIR / 'bin' / 'inputs' / 'nbm_param_sc.txt',
        'rb': cfg.MODEL_DIR / 'bin' / 'inputs' / 'nbm_param_rb.txt',
    }

    nbm_param_file = config.get('fragmentation', 'nbm_param_file')
    nbm_param_default = config.get('fragmentation', 'nbm_param_default')
    nbm_param_type = config.get('fragmentation', 'nbm_param_type')

    if nbm_param_file is None:
        if nbm_param_default is None:
            raise ValueError('No nasa-breakup-model parameter file given')
        nbm_param_file = str(nbm_param_defaults[nbm_param_default])

    fragmentation_time = config.getfloat('fragmentation', 'fragmentation_time')

    cache_file = fragmentation_output / 'nbm_cloud_data.pickle'
    if config.getboolean('general', 'use-cache') and cache_file.is_file():
        logger.info('Using nbm cloud cache')
        with open(cache_file, 'rb') as fh:
            cloud_data, header = pickle.load(fh)
    else:
        logger.info(f'Using nbm file: {nbm_param_file}')
        cloud_data, header = ndm_wrapper.run_nasa_breakup(
            nbm_param_file, 
            nbm_param_type, 
            space_object, 
            fragmentation_time,
        )
        with open(cache_file, 'wb') as fh:
            pickle.dump([cloud_data, header], fh)

        nbm_folder = fragmentation_output / 'nbm_results'
        nbm_folder.mkdir(parents=True, exist_ok=True)
        nbm_results = (cfg.MODEL_DIR / 'bin' / 'outputs').glob('*.txt')
        for file in nbm_results:
            logger.info(f'Moving nasa-breakup-model result to output {nbm_folder / file.name}')
            shutil.copy(str(file), str(nbm_folder / file.name))

        ndm_plotting.plot(cfg.MODEL_DIR / 'bin' / 'outputs', nbm_folder / 'plots')

    fdt = TimeDelta(fragmentation_time*3600.0, format='sec')
    fragmentation_epoch = space_object.epoch + fdt

    tracklet_point_spacing = 10.0

    scheduler_modes = [None]
    custom_scheduler_getter = get_custom_scheduler_getter(config, output)
    if custom_scheduler_getter is not None:
        scheduler_modes.append(custom_scheduler_getter)

    for mode in scheduler_modes:
        if mode is not None:
            logger.info('Running custom scheduler scheduler')
            file_modifier = 'custom_scheduler_'
            _radar = mode().radar
        else:
            logger.info('Running optimal tracking scheduler')
            file_modifier = ''
            _radar = radar

        cache_file = fragmentation_output / f'{file_modifier}fragment_pass_data.pickle'
        if config.getboolean('general', 'use-cache') and cache_file.is_file():
            logger.info('Using nbm fragment observation cache')
            with open(fragmentation_output / f'{file_modifier}fragment_pass_data.pickle', 'rb') as fh:
                fragment_pass_data = pickle.load(fh)
            with open(fragmentation_output / f'{file_modifier}fragment_data.pickle', 'rb') as fh:
                fragment_data = pickle.load(fh)
            fragment_states = np.load(fragmentation_output / f'{file_modifier}fragment_states.npz')

        else:
            logger.info('Simulating fragment observation')

            fragment_pass_data, fragment_data, fragment_states = sst_simulation.observe_nbm_fragments(
                fragmentation_epoch, 
                cloud_data, 
                _radar,
                tracklet_point_spacing, 
                obs_epoch,
                t,
                propagator, 
                propagator_options, 
                cores,
                custom_scheduler_getter=mode,
            )

            with open(fragmentation_output / f'{file_modifier}fragment_pass_data.pickle', 'wb') as fh:
                pickle.dump(fragment_pass_data, fh)
            with open(fragmentation_output / f'{file_modifier}fragment_data.pickle', 'wb') as fh:
                pickle.dump(fragment_data, fh)
            np.savez(fragmentation_output / f'{file_modifier}fragment_states.npz', **fragment_states)

        if mode is None:
            if config.getboolean('fragmentation', 'animate'):
                output_anim = fragmentation_output / 'animation'
                output_anim.mkdir(exist_ok=True)
                plotting.animate_fragments(t, fragment_states, output_anim, cores=cores)

        figsize = (
            config.custom_getint('general', 'figsize-x'),
            config.custom_getint('general', 'figsize-y'),
        )
        figsize = tuple([x for x in figsize if x is not None])
        if len(figsize) < 2:
            figsize = None

        fig_frags, axes = plotting.fragment_stats(_radar, fragment_pass_data, figsize=figsize)
        fig_frags.savefig(fragmentation_output / f'{file_modifier}fragments_snr_distributions.png')

        fig_frags, axes = plotting.fragment_stats(_radar, fragment_pass_data, obs_epoch=obs_epoch, figsize=figsize)
        fig_frags.autofmt_xdate()
        fig_frags.savefig(fragmentation_output / f'{file_modifier}fragments_snr_distributions_epoch.png')


def run_orbit_planner(args, config, cores, radar, output, CACHE, profiler=None):

    space_object, propagator, propagator_options = CACHE.get_cache(
        'space_object',
        lambda: get_base_object(config, args),
    )
    t_data = CACHE.get_cache(
        't_data',
        lambda: get_times(config, space_object),
    )
    obs_epoch, t, t_start, t_end, t_step = t_data

    orbit_output = output / 'sampled_orbit'
    orbit_output.mkdir(parents=True, exist_ok=True)

    __out_frame = space_object.out_frame
    space_object.out_frame = 'TEME'
    orbit_state = space_object.get_state([0.0])
    space_object.out_frame = __out_frame

    scheduler_modes = [None]
    custom_scheduler_getter = get_custom_scheduler_getter(config, output)
    if custom_scheduler_getter is not None:
        scheduler_modes.append(custom_scheduler_getter)

    for mode in scheduler_modes:
        if mode is not None:
            logger.info('Running custom scheduler scheduler')
            file_modifier = 'custom_scheduler_'
            _radar = mode().radar
        else:
            logger.info('Running optimal tracking scheduler')
            file_modifier = ''
            _radar = radar
        
        cache_file = orbit_output / f'{file_modifier}orbit_data.pickle'
        if config.getboolean('general', 'use-cache') and cache_file.is_file():
            logger.info('Using orbit samples cache')
            with open(cache_file, 'rb') as fh:
                orbit_samples_data = pickle.load(fh)
        else:
            logger.info('Processing orbit samples')
            orbit_samples_data = sst_simulation.process_orbit(
                orbit_state,
                space_object.parameters,
                space_object.epoch,
                t,
                propagator,
                propagator_options,
                radar,
                profiler,
                obs_epoch,
                config.getint('orbit', 'samples'),
                cores=cores,
                custom_scheduler_getter=mode,
            )
            with open(cache_file, 'wb') as fh:
                pickle.dump(orbit_samples_data, fh)

        figsize = (
            config.custom_getint('general', 'figsize-x'),
            config.custom_getint('general', 'figsize-y'),
        )
        figsize = tuple([x for x in figsize if x is not None])
        if len(figsize) < 2:
            figsize = None

        fig_orb, axes = plotting.orbit_sampling(_radar, orbit_samples_data, obs_epoch=obs_epoch, figsize=figsize)
        fig_orb.autofmt_xdate()
        fig_orb.savefig(orbit_output / f'{file_modifier}orbit_sampling_max_snr_epoch.png')

        fig_orb, axes = plotting.orbit_sampling(_radar, orbit_samples_data, figsize=figsize)
        fig_orb.savefig(orbit_output / f'{file_modifier}orbit_sampling_max_snr.png')

        fig_orb, axes = plotting.orbit_sampling(_radar, orbit_samples_data, snr_mode='all', figsize=figsize)
        fig_orb.savefig(orbit_output / f'{file_modifier}orbit_sampling_snr.png')

        fig_orb, axes = plotting.orbit_sampling(_radar, orbit_samples_data, snr_mode='optimal', figsize=figsize)
        fig_orb.savefig(orbit_output / f'{file_modifier}orbit_sampling_best_snr.png')

        color_set = sorts.plotting.colors.get_cset('muted')

        _t, sn, az, el = plotting.publish_orbit_sampling(_radar, orbit_samples_data, obs_epoch=obs_epoch)
        fig_orb, axes = plt.subplots(2, len(_radar.rx), figsize=figsize, sharex=True)
        axes = axes.T.reshape((len(radar.rx), 2))
        for rxi in range(len(_radar.rx)):
            axes[rxi-1][0].plot(_t, 10*np.log10(sn), '-k')
            axes[rxi-1][0].set_ylabel('SNR [dB]')

            snm = np.argmax(sn)
            axes[rxi-1][0].axvline(_t[snm], color=color_set.wine)

            axes[rxi-1][1].plot(_t, az, '-', color=color_set.purple)
            axes[rxi-1][1].tick_params(axis='y', labelcolor=color_set.purple)
            axes[rxi-1][1].set_ylabel('Azimuth [deg]', color=color_set.purple)

            ax2 = axes[rxi-1][1].twinx()
            ax2.set_ylabel('Elevation [deg]', color=color_set.green)  # we already handled the x-label with ax1
            ax2.plot(_t, el, '-.', color=color_set.green)
            ax2.tick_params(axis='y', labelcolor=color_set.green)
            
            axes[rxi-1][1].axvline(_t[snm], color=color_set.wine, label='Peak SNR')
            axes[rxi-1][1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d %H:%M'))
            axes[rxi-1][1].locator_params(axis='x', nbins=5)
            axes[rxi-1][1].grid(True)
        
        fig_orb.suptitle('Orbit max SNR prediction')
        fig_orb.autofmt_xdate()

        fig_orb.savefig(orbit_output / f'{file_modifier}orbit_max_snr_epoch.png')



def get_base_object(config, args):
    
    if args.propagator.lower().startswith('sgp4'):
        propagator = sorts.propagator.SGP4
        propagator_options = dict(
            settings=dict(
                in_frame='TEME',
                out_frame='ITRS',
                tle_input=False,
            )
        )
    elif args.propagator.lower() == 'kepler':
        propagator = sorts.propagator.Kepler
        propagator_options = dict(
            settings=dict(
                in_frame='TEME',
                out_frame='ITRS',
            )
        )
    elif args.propagator.lower() == 'orekit':
        orekit_data = config.get('general', 'orekit-data')
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
    logger.info(f'Using propagator: {args.propagator}')

    line1 = config.get('orbit', 'line 1')
    line2 = config.get('orbit', 'line 2')

    parameters = dict(
        d = config.custom_getfloat('orbit', 'd'),
        A = config.custom_getfloat('orbit', 'a'),
        m = config.custom_getfloat('orbit', 'm'),
        B = config.custom_getfloat('orbit', 'B'),
    )
    parameters = {key: val for key, val in parameters.items() if val is not None}

    if line1 is not None and line2 is not None:
        logger.info('Using TLEs')
        space_object_tle = sst_simulation.get_space_object_from_tle(
            line1, 
            line2, 
            parameters,
        )
        if args.propagator.lower() == 'sgp4':
            space_object = space_object_tle
        elif args.propagator.lower() == 'sgp4-state':
            space_object = sst_simulation.convert_tle_so_to_state_so(
                space_object_tle, 
                propagator = propagator,
                propagator_options = propagator_options,
                samples=500, 
            )
        else:
            space_object = sst_simulation.convert_tle_so_to_state_so(
                space_object_tle, 
                propagator = propagator,
                propagator_options = propagator_options,
                samples=1, 
            )
    else:
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
        logger.info(f'Using {coords} state')

        space_object = sorts.SpaceObject(
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
    
    logger.info(space_object)

    return space_object, propagator, propagator_options


def get_custom_scheduler_getter(config, output):

    custom_scheduler_file = config.getboolean('general', 'scheduler-file')

    if custom_scheduler_file is None or not custom_scheduler_file:
        custom_scheduler_getter = None
        logger.info('No custom scheduler, using optimal SNR tracking')
    else:
        import custom_scheduler
        custom_scheduler_getter = custom_scheduler.get_scheduler
        logger.info('Custom scheduler loaded, adding observation prediction')
        logger.info(custom_scheduler_getter)

    # this cannot be pickled?
    # -----
    # if custom_scheduler_file is not None:
    #     scheduler_file_path = pathlib.Path(custom_scheduler_file)
    #     spec = importlib.util.spec_from_file_location(
    #         "custom_scheduler_module",
    #         str(scheduler_file_path.resolve()),
    #     )
    #     custom_scheduler_module = importlib.util.module_from_spec(spec)
    #     spec.loader.exec_module(custom_scheduler_module)
    #     custom_scheduler_getter = custom_scheduler_module.get_scheduler
    #     logger.info('Custom scheduler loaded, adding observation prediction')
    #     logger.info(custom_scheduler_getter)

    #     shutil.copy(scheduler_file_path, output / scheduler_file_path.name)
    #     logger.info(f'Creating copy of scheduler: {output / scheduler_file_path.name}')
    # else:
    #     custom_scheduler_getter = None
    #     logger.info('No custom scheduler, using optimal SNR tracking')

    return custom_scheduler_getter


def get_times(config, space_object):

    obs_epoch = config.custom_getfloat('general', 'epoch')
    if obs_epoch is None:
        obs_epoch = space_object.epoch
    else:
        obs_epoch = Time(obs_epoch, format='mjd')

    logger.info(f'Planning epoch = {obs_epoch}')
    t_start = config.getfloat('general', 't_start')*3600.0
    t_end = config.getfloat('general', 't_end')*3600.0
    t_step = config.getfloat('general', 't_step')

    logger.info(f'Planning time = t0 + {t_start/3600.0} h -> t0 + {t_end/3600.0} h')
    
    t = np.arange(t_start, t_end, t_step)

    return obs_epoch, t, t_start, t_end, t_step


if __name__ == '__main__':
    profiler.start('total')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    logger_info = 'Avalible loggers: \n'
    for name in logging.root.manager.loggerDict:
        if name.startswith('sst-cli'):
            logger_info += f'- {name}\n'
    logger.info(logger_info)

    actions = {
        'passes': run_passes_planner,
        'object': run_object_planner, 
        'orbit': run_orbit_planner,
        'fragmentation': run_fragmentation_planner, 
    }
    action_names = list(actions.keys())

    parser = argparse.ArgumentParser(description='\
        Plan SSA observations of a single object, \
        fragmentation event or of an entire orbit')
    parser.add_argument('radar', type=str, help='The observing radar system')
    parser.add_argument('config', type=str, help='Config file for the planning')
    parser.add_argument('output', type=str, help='Path to output results')
    parser.add_argument('propagator', type=str, help='Propagator to use')
    parser.add_argument(
        '--target',
        default=['passes'],
        choices=action_names,
        nargs='+',
        help='Type of target for observation',
    )
    parser.add_argument('--txi', type=int, default=[0], nargs='?', help='TX indecies to use')
    parser.add_argument('--rxi', type=int, default=[0], nargs='?', help='RX indecies to use')

    args = parser.parse_args()

    radar = getattr(sorts.radars, args.radar)

    radar.tx = [tx for txi, tx in enumerate(radar.tx) if txi in args.txi]
    radar.rx = [rx for rxi, rx in enumerate(radar.rx) if rxi in args.rxi]

    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    conf_path = pathlib.Path(args.config)
    config = cfg.get_config(conf_path)

    shutil.copy(args.config, output / conf_path.name)

    with open(output / 'cmd.txt', 'w') as fh:
        fh.write(" ".join(sys.argv))

    cores = config.custom_getint('general', 'cores')
    if cores is None or cores < 1:
        cores = 1

    # This is for avoiding recomputation
    CACHE = delayed_setdefault()

    for action in action_names:
        if action not in args.target:
            continue

        func = actions[action]
        if func is None:
            raise NotImplementedError(f'Not enough coffee, "{action}" not implemented')

        profiler.start(f'{action}')
        func(args, config, cores, radar, output, CACHE, profiler=profiler)
        profiler.stop(f'{action}')

    profiler.stop('total')
    logger.info(profiler.fmt(normalize='total'))
