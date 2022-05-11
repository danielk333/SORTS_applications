#!/usr/bin/env python

'''
NEO detection simulation

Can be used as

python main.py -s 86.4 -g 0.14 "2004 MN4" "eiscat_3d" "2029-04-09T00:00:00" 8.0 \
    "/home/danielk/data/jpl_eph/de430_plus_MarsPC.bsp" "./results/2004_MN4_3d/"
python main.py -s 86.4 -g 0.14 "2004 MN4" "eiscat_uhf" "2029-04-09T00:00:00" 8.0 \
    "/home/danielk/data/jpl_eph/de430_plus_MarsPC.bsp" "./results/2004_MN4_uhf/"
python main.py -s 10000.0 -g 0.14 "1994 PC1" "eiscat_uhf" "2022-01-14T00:00:00" 8.0 \
    "/home/danielk/data/jpl_eph/de430_plus_MarsPC.bsp" "./results/1994_PC1/"

python main.py --full-fov -s 10000.0 -g 0.14 --stations 0 0 "2010 XC15" "eiscat_uhf" \
    "2022-12-15T00:00:00" 20.0 \
    "/home/danielk/data/jpl_eph/de430_plus_MarsPC.bsp" \
    "./results/2010_XC15_uhf_nofov/"
python main.py -s 10000.0 -g 0.14 --stations 0 0 "2010 XC15" "eiscat_uhf" \
    "2022-12-15T00:00:00" 20.0 \
    "/home/danielk/data/jpl_eph/de430_plus_MarsPC.bsp" \
    "./results/2010_XC15_uhf/"
python main.py --full-fov -s 10000.0 -g 0.14 --stations 0 0 "2010 XC15" "eiscat_esr" \
    "2022-12-15T00:00:00" 20.0 \
    "/home/danielk/data/jpl_eph/de430_plus_MarsPC.bsp" \
    "./results/2010_XC15_esr_nofov/"
python main.py -s 10000.0 -g 0.14 --stations 0 0 "2010 XC15" "eiscat_esr" \
    "2022-12-15T00:00:00" 20.0 \
    "/home/danielk/data/jpl_eph/de430_plus_MarsPC.bsp" \
    "./results/2010_XC15_esr/"

'''
import pathlib
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astroquery.jplhorizons import Horizons

import sorts
import pyorb

from schedulers import TrackingScheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plan NEO observations')
    parser.add_argument(
        'target', type=str, 
        help='Target object with id according to JPL Horizons',
    )
    parser.add_argument(
        'radar', type=str, 
        help='The observing radar system',
    )
    parser.add_argument(
        'start_date', type=str, 
        help='Simulation starting date and time [ISO]',
    )
    parser.add_argument(
        'sim_time', type=float, 
        help='Number of days to simulate [days]',
    )
    parser.add_argument(
        'kernel', type=str, 
        help='The JPL Kernel used for solar-system simulation',
    )
    parser.add_argument(
        'figure_output', type=str, 
        help='Path to output figures',
    )
    parser.add_argument(
        '--time_step', type=float, default=3600.0, 
        help='Propagation output separation [sec]',
    )
    parser.add_argument(
        '--time_slice', type=float, default=3600.0, 
        help='SNR Integration time [sec]',
    )
    parser.add_argument(
        '-s', '--spin_period', type=float, default=86.4, 
        help='Spin period of the target object (affects SNR) [sec]',
    )
    parser.add_argument(
        '-g', '--geometric_albedo', type=float, default=0.14, 
        help='Geometric albedo of the object',
    )
    parser.add_argument(
        '--full-fov', action='store_true',
        help='Do not use elevation limits',
    )
    parser.add_argument(
        '--stations', type=int, nargs=2, default=[0, 0],
        help='TX station and RX station index of the radar',
    )

    args = parser.parse_args()

    radar = getattr(sorts.radars, args.radar)

    if args.full_fov:
        for st in radar.tx + radar.rx:
            st.min_elevation = 0

    radar.tx = [radar.tx[args.stations[0]]]
    radar.rx = [radar.rx[args.stations[1]]]

    dt = args.time_step
    days = args.sim_time
    LD = 384402e3
    t_slice = args.time_slice
    geometric_albedo = args.geometric_albedo

    kernel = pathlib.Path(args.kernel)

    figure_output = pathlib.Path(args.figure_output)
    figure_output.mkdir(parents=True, exist_ok=True)

    assert kernel.is_file(), 'Kernel does not exist'

    epoch = Time(args.start_date, format='isot', scale='tdb')

    propagator_options = dict(
        kernel = kernel, 
        settings=dict(
            in_frame='HeliocentricMeanEcliptic',
            out_frame='ITRS',
            time_step = 30.0, 
            save_massive_states = True, 
            epoch_scale = 'tdb',
        ),
    )

    jpl_obj = Horizons(
        id=args.target, 
        location='500@10', 
        epochs=epoch.jd,
    )
    jpl_el = jpl_obj.elements()
    print(jpl_obj)
    print(jpl_el)
    for key in jpl_el.keys():
        print(f'{key}:{jpl_el[key].data[0]}')

    orb = pyorb.Orbit(
        M0 = pyorb.M_sol, 
        direct_update=True, 
        auto_update=True, 
        degrees=True, 
        a=jpl_el['a'].data[0]*pyorb.AU, 
        e=jpl_el['e'].data[0], 
        i=jpl_el['incl'].data[0], 
        omega=jpl_el['w'].data[0], 
        Omega=jpl_el['Omega'].data[0], 
        anom=jpl_el['M'].data[0],
        type='mean',
    )
    print('Initial orbit:')
    print(orb)

    def H_to_D(H, pV):
        return 10**(3.1236 - 0.5*np.log10(pV) - 0.2*H)

    diameter = H_to_D(jpl_el['H'].data[0], geometric_albedo)*1e3
    print(f'ESTIMATED DIAMETER: {diameter} m')

    obj = sorts.SpaceObject(
        sorts.propagator.Rebound,
        propagator_options = propagator_options,
        state = orb,
        epoch = epoch,
        parameters = dict(
            H = jpl_el['H'].data[0],
            d = diameter,
            geometric_albedo = geometric_albedo,
            spin_period = args.spin_period,
        ),
    )
    print(obj)

    t = np.arange(0, 3600.0*24.0*days, dt)

    states, massive_states = obj.get_state(t)
    interpolator = sorts.interpolation.Legendre8(states, t)

    scheduler = TrackingScheduler(radar, t, states, t_slice=t_slice)

    for tx_p in scheduler.passes:
        for rx_p in tx_p:
            for ps in rx_p:
                print(ps)

    data = scheduler.observe_passes(
        scheduler.passes, 
        space_object=obj, 
        epoch=epoch, 
        doppler_spread_integrated_snr=True,
        interpolator=interpolator, 
        snr_limit=False,
        extended_meta=False,
    )

    # plot results
    fig1 = plt.figure(figsize=(15, 15))
    axes = [
        fig1.add_subplot(131),
    ]
    r_axes = [
        fig1.add_subplot(132),
    ]
    sn_axes = [
        fig1.add_subplot(133),
    ]

    for tx_d in data:
        for rxi, rx_d in enumerate(tx_d):
            for dati, dat in enumerate(rx_d):
                axes[rxi].plot(
                    dat['tx_k'][0, :], dat['tx_k'][1, :], 
                    label=f'Pass {dati}',
                )
                sn_axes[rxi].plot(
                    (dat['t'] - np.min(dat['t']))/(3600.0*24), 
                    10*np.log10(dat['snr']), label=f'Pass {dati}',
                )
                r_axes[rxi].plot(
                    (dat['t'] - np.min(dat['t']))/(3600.0*24), 
                    (dat['range']*0.5)/LD, 
                    label=f'Pass {dati}',
                )

    axes[0].legend()
    for rxi, ax in enumerate(axes):
        ax.set_xlabel('k_x [East]')
        ax.set_ylabel('k_y [North]')
        ax.set_title(f'Receiver station {rxi}')

    for rxi, ax in enumerate(sn_axes):
        ax.set_xlabel('Time during pass [d]')
        ax.set_ylabel('SNR [dB/h]')
        ax.set_title(f'Receiver station {rxi}')

    r_axes[0].legend()
    for rxi, ax in enumerate(r_axes):
        ax.set_xlabel('Time during pass [d]')
        ax.set_ylabel('Range [LD]')
        ax.set_title(f'Receiver station {rxi}')

    # plot results
    fig3 = plt.figure(figsize=(15, 15))
    ax = fig3.add_subplot(111, projection='3d')
    ax.plot(states[0, :]/LD, states[1, :]/LD, states[2, :]/LD, 'b')
    for ctrl in scheduler.controllers:
        for ti in range(len(ctrl.t)):
            ax.plot(
                [radar.tx[0].ecef[0]/LD, ctrl.ecefs[0, ti]/LD], 
                [radar.tx[0].ecef[1]/LD, ctrl.ecefs[1, ti]/LD], 
                [radar.tx[0].ecef[2]/LD, ctrl.ecefs[2, ti]/LD], 
                'g',
            )

    fig4 = plt.figure(figsize=(15, 15))
    ax = fig4.add_subplot(111)
    ax.plot(t/(3600.0*24), np.linalg.norm(states[:3, :], axis=0)/LD)

    ax.set_xlabel('Time since epoch [d]')
    ax.set_ylabel('Distance from Earth [LD]')

    fig1.savefig(figure_output / 'results.png')
    fig3.savefig(figure_output / '3d_obs.png')
    fig4.savefig(figure_output / 'earth_distance.png')

    with open(figure_output / 'cmd.txt', 'w') as fh:
        fh.write(" ".join(sys.argv))
