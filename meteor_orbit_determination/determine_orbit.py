#!/usr/bin/env python

'''
Calculating distributions pre-encounter orbits
=================================================

'''

import pathlib
import datetime
import argparse

import numpy as np
from numpy.random import default_rng
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from astropy.time import Time, TimeDelta

import sorts
import pyorb


def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.

    Source: https://gist.github.com/victorkristof/b9d794fe1ed12e708b9d
    
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    dt = datetime.datetime.fromordinal(int(datenum))
    dt += datetime.timedelta(days=days)
    dt -= datetime.timedelta(days=366)
    return dt


def propagate_pre_encounter(
                state, epoch, in_frame, out_frame, kernel, 
                dt = 10.0, t_target = -10*3600.0, t_step = -60.0, 
                settings = None,
            ):

    t = np.arange(0, t_target + t_step, t_step)

    reb_settings = dict(
        in_frame=in_frame,
        out_frame=out_frame,
        time_step = 10.0,  # s
        termination_check = False,
        save_massive_states = True,
    )
    if settings is not None:
        settings.update(reb_settings)
    else:
        settings = reb_settings

    prop = sorts.propagator.Rebound(
        kernel = kernel, 
        settings = settings,
    )

    states, massive_states = prop.propagate(t, state, epoch)
    earth_index = prop.planet_index('Earth')

    return states, massive_states, t, earth_index


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Determine meteor orbit')
    parser.add_argument('input', type=str, help='Input states (h5 format)')
    parser.add_argument('lat', type=float, help='The observing radar system latitude')
    parser.add_argument('lon', type=float, help='The observing radar system longitude')
    parser.add_argument('alt', type=float, help='The observing radar system altitude')
    parser.add_argument('kernel', type=str, help='The JPL Kernel used for solar-system simulations')
    parser.add_argument('samples', type=int, help='Samples to pick from all states, <= 0 as well as same size as input implies all samples')
    parser.add_argument('output', type=str, help='Path to output data (h5 format)')
    parser.add_argument('-p', '--plot', type=str, default='', help='Plot results')
    parser.add_argument('--time-series', type=str, default='', help='Time series cache (npz format)')
    parser.add_argument('-c', '--clobber', action='store_true', help='Override output')

    args = parser.parse_args()

    with h5py.File(args.input, 'r') as h:
        # saved transposed
        states = h['sample_states'][()].T
        num = states.shape[1]
        
        # thinning
        if args.samples > 0 and args.samples < num:
            rng = default_rng()
            inds = rng.choice(states.shape[1], size=args.samples, replace=False)
            states = states[:, inds]
        else:
            args.samples = num
        
        datetime_epoch = datenum_to_datetime(h.attrs['epoch'][0])
        epoch = Time(datetime_epoch, scale='utc', format='datetime')

    states_mu = np.mean(states, axis=1)

    print(f'Detection epoch: {epoch.iso}')
    print(f'Input states: {num}')
    print(f'Thinned to: {args.samples}')
    print(f'Using JPL kernel: {args.kernel}')

    radar_itrs = sorts.frames.geodetic_to_ITRS(args.lat, args.lon, args.alt)
    states_ITRS = states.copy()
    states_ITRS[:3, :] = radar_itrs[:, None] + sorts.frames.enu_to_ecef(args.lat, args.lon, args.alt, states[:3, :])
    states_ITRS[3:, :] = sorts.frames.enu_to_ecef(args.lat, args.lon, args.alt, states[3:, :])

    ts_file = pathlib.Path(args.time_series)
    if not ts_file.is_file() or args.clobber:
        states_prop, massive_states, t, earth_index = propagate_pre_encounter(
            states_ITRS, 
            epoch, 
            in_frame = 'ITRS', 
            out_frame = 'HCRS', 
            kernel = args.kernel, 
            settings = dict(tqdm=True),
        )
        earth_state = np.squeeze(massive_states[:3, :, earth_index])
        d_earth = np.linalg.norm(states_prop[:3, :, :] - earth_state[:, :, None], axis=0)/pyorb.AU
        if len(args.time_series) > 0:
            np.savez(ts_file, states_prop=states_prop, massive_states=massive_states, t=t, d_earth=d_earth)
    else:
        dat = np.load(ts_file)
        states_prop = dat['states_prop']
        massive_states = dat['massive_states']
        t = dat['t']
        d_earth = dat['d_earth']

    out_file = pathlib.Path(args.output)
    if not out_file.is_file() or args.clobber:
        states_end = np.squeeze(states_prop[:, -1, :])

        end_epoch = epoch + TimeDelta(t[-1], format='sec')
        end_mjd = end_epoch.mjd

        states_HMC = sorts.frames.convert(
            end_epoch,
            states_end, 
            in_frame = 'HCRS', 
            out_frame = 'HeliocentricMeanEcliptic',
        )

        orb = pyorb.Orbit(
            M0 = pyorb.M_sol,
            direct_update=True,
            auto_update=True,
            degrees = True,
            num = args.samples,
        )
        orb.cartesian = states_HMC
        kepler = orb.kepler

        with h5py.File(out_file, 'w') as h:
            h.create_dataset('kepler', data=orb.kepler)
            h.create_dataset('states', data=orb.cartesian)
            h.attrs['mjd'] = end_mjd
    else:
        with h5py.File(out_file, 'r') as h:
            kepler = h['kepler'][()]

    if len(args.plot) > 0:
        plot_path = pathlib.Path(args.plot)
        if not plot_path.is_dir():
            plot_path = None
    else:
        plot_path = None

    if plot_path is not None:
        print('Plotting...')
        plt.rc('text', usetex=True)

        axis_labels = [
            "$a$ [AU]", "$e$ [1]", "$i$ [deg]", 
            "$\\omega$ [deg]", "$\\Omega$ [deg]", "$\\nu$ [deg]"
        ]
        scale = [1/pyorb.AU] + [1]*5

        fig = plt.figure(figsize=(15, 15))
        for dim in range(6):
            ax = fig.add_subplot(231+dim)
            ax.hist(kepler[dim, :]*scale[dim])
            ax.set_ylabel('Frequency')
            ax.set_xlabel(axis_labels[dim])
            if dim == 4:
                ax.ticklabel_format(axis='x', useOffset=False)
        fig.suptitle(f'Elements at pre-encounter {t[-1]/3600} h')

        fig.savefig(plot_path / 'final_kepler.png')

        fig, ax = plt.subplots(figsize=(8, 8))
        for ind in range(args.samples):
            ax.plot(t/3600.0, d_earth[:, ind], color='k', alpha=0.1)
        ax.axhline(0.01, color='r')
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Distance from Earth [AU]')

        fig.savefig(plot_path / 'earth_distance.png')

        with h5py.File(out_file, 'a') as h:
            if 'all_kepler' not in h or args.clobber:
                orb_t = pyorb.Orbit(
                    M0=pyorb.M_sol,
                    direct_update=True,
                    auto_update=True,
                    degrees=True,
                    num=args.samples*len(t),
                )
                t_mat = np.tile(t.reshape((len(t), 1)), (1, args.samples))
                tv = epoch + TimeDelta(t_mat.reshape((args.samples*len(t),)), format='sec')
                orb_t.cartesian = sorts.frames.convert(
                    tv,
                    states_prop.reshape((6, args.samples*len(t))), 
                    in_frame = 'HCRS', 
                    out_frame = 'HeliocentricMeanEcliptic',
                )
                all_kepler = orb_t.kepler.reshape((6, len(t), args.samples))
                del orb_t
                if 'all_kepler' in h:
                    del h['all_kepler']
                h.create_dataset('all_kepler', data=all_kepler)
            else:
                with h5py.File(out_file, 'r') as h:
                    all_kepler = h['all_kepler'][()]

        print('Kepler calculation done...')

        all_kepler[0, :, :] = 1.0/(all_kepler[0, :, :]/pyorb.AU)
        scale[0] = 1.0
        axis_labels[0] = "$a^{-1}$ [AU$^{-1}$]"

        fig, axes = plt.subplots(2, 3, figsize=(15, 15))
        axes = axes.flatten()

        histogram_bins = int(np.sqrt(args.samples)*0.5)
        bins = [
            np.linspace(np.amin(all_kepler[dim, :, :]), np.amax(all_kepler[dim, :, :]), histogram_bins + 1)
            for dim in range(6)
        ]
        bin_mids = [
            0.5*(b[1:] + b[:-1])
            for b in bins
        ]

        time_kep = np.empty((6, len(t), histogram_bins), dtype=states_prop.dtype)

        for ind in range(len(t)):
            for dim in range(6):
                time_kep[dim, ind, :], _ = np.histogram(
                    all_kepler[dim, ind, :], 
                    bins=bins[dim], 
                    density=True,
                )

        logt = np.log10(np.abs(t[1:]/3600.0))
        t_passed = logt[np.argmax(np.min(d_earth[1:, :], axis=1) > 0.01)]

        for dim in range(6):
            T, K = np.meshgrid(logt, bin_mids[dim]*scale[dim])
            cs = axes[dim].pcolormesh(T, K, np.squeeze(time_kep[dim, 1:, :]).T, cmap='binary')
            axes[dim].set_xlim(logt.min(), logt.max())
            axes[dim].invert_xaxis()
            ticks_loc = axes[dim].get_xticks().tolist()
            new_locs = [f'-{10**loc}' for loc in ticks_loc]
            axes[dim].set_xticks(ticks_loc)
            axes[dim].set_xticklabels(new_locs)
            if dim in [2, 5]:
                cb = fig.colorbar(cs, ax=axes[dim])
                cb.set_label('Probability [1]')
            if dim == 4:
                axes[dim].ticklabel_format(axis='y', useOffset=False)

            axes[dim].axvline(t_passed, color='r')

        for dim in range(6):
            axes[dim].set_xlabel('Time before epoch [h]')
            axes[dim].set_ylabel(axis_labels[dim])
        fig.suptitle('Propagation to pre-encounter elements')

        fig.savefig(plot_path / 'kepler_vs_t.png')

        plt.show()
