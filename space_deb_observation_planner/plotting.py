import logging
import multiprocessing as mp
import subprocess

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import TimeDelta

import sorts
import pyant

logger = logging.getLogger('sst-cli.plotting')


def orbit_sampling(radar, orbit_samples_data, snr_mode='max', obs_epoch=None, figsize=None):

    txi = 0
    optimal_dt = 3*10.0  # sec

    fig_orb = plt.figure(figsize=figsize)
    axes = []
    for rxi in range(1, len(radar.rx)+1):
        axes.append([])
        axes[rxi-1].append(fig_orb.add_subplot(3, len(radar.rx), 0*len(radar.rx)+rxi))
        axes[rxi-1].append(fig_orb.add_subplot(3, len(radar.rx), 1*len(radar.rx)+rxi))
        axes[rxi-1].append(fig_orb.add_subplot(3, len(radar.rx), 2*len(radar.rx)+rxi))

        t = []
        sn = []
        az = []
        el = []
        for oid in range(len(orbit_samples_data)):
            for ind, d in enumerate(orbit_samples_data[oid][txi][rxi-1]):
                if snr_mode == 'max':
                    snm = np.argmax(d['snr'])
                    sn.append(d['snr'][snm])
                    t.append(d['t'][snm])
                    azel = pyant.coordinates.cart_to_sph(d['rx_k'][:, snm], radians=False)
                    az.append(azel[0])
                    el.append(azel[1])
                elif snr_mode in ['all', 'optimal']:
                    sn.append(d['snr'].copy())
                    t.append(d['t'].copy())
                    azel = pyant.coordinates.cart_to_sph(d['rx_k'], radians=False)
                    az.append(azel[0, ...])
                    el.append(azel[1, ...])
                else:
                    raise TypeError(snr_mode)
        if snr_mode == 'max':
            t = np.array(t)
            sn = np.array(sn)
            az = np.array(az)
            el = np.array(el)
        elif snr_mode in ['all', 'optimal']:
            t = np.concatenate(t)
            sn = np.concatenate(sn)
            az = np.concatenate(az)
            el = np.concatenate(el)

        t_sort = np.argsort(t)
        t = t[t_sort]
        sn = sn[t_sort]
        az = az[t_sort]
        el = el[t_sort]

        if snr_mode == 'optimal':
            t_optimize = np.arange(t[0], t[-1] + optimal_dt, optimal_dt, dtype=np.float64)
            t_new = np.empty((len(t_optimize) - 1,), dtype=np.float64)
            sn_new = np.empty_like(t_new)
            az_new = np.empty_like(t_new)
            el_new = np.empty_like(t_new)
            for ind in range(len(t_new)):
                selector = np.logical_and(t >= t_optimize[ind], t <= t_optimize[ind+1])
                if np.any(selector):
                    sn_max = np.argmax(sn[selector])
                    t_new[ind] = t[selector][sn_max]
                    sn_new[ind] = sn[selector][sn_max]
                    az_new[ind] = az[selector][sn_max]
                    el_new[ind] = el[selector][sn_max]
                else:
                    t_new[ind] = np.nan
                    sn_new[ind] = np.nan
                    az_new[ind] = np.nan
                    el_new[ind] = np.nan

            t = t_new
            sn = sn_new
            az = az_new
            el = el_new

        if obs_epoch is None:
            _t = t/3600.0
        else:
            _t = (obs_epoch + TimeDelta(t, format='sec')).datetime

        axes[rxi-1][0].plot(_t, 10*np.log10(sn), '.b')
        axes[rxi-1][0].set_ylabel('Orbit sample SNR [dB]')
        axes[rxi-1][0].set_xlabel('Time past epoch [h]')

        axes[rxi-1][1].plot(_t, az, '.b')
        axes[rxi-1][1].set_ylabel('Azimuth [deg]')
        axes[rxi-1][1].set_xlabel('Time past epoch [h]')

        axes[rxi-1][2].plot(_t, el, '.b')
        axes[rxi-1][2].set_ylabel('Elevation [deg]')
        axes[rxi-1][2].set_xlabel('Time past epoch [h]')

    return fig_orb, axes


def fragment_stats(radar, fragment_pass_data, obs_epoch=None, figsize=None):

    fig_frags, axes = plt.subplots(3, len(radar.rx), figsize=figsize)
    if len(axes.shape) == 1:
        axes.shape = (3, 1)

    for ind in range(0, len(radar.rx)):

        sn = np.array(fragment_pass_data[0][ind]['peak_snr'])
        t = np.array(fragment_pass_data[0][ind]['peak_time'])

        if obs_epoch is not None:
            t = (obs_epoch + TimeDelta(t, format='sec')).datetime
        else:
            t = t/3600.0

        ax = axes[0, ind]
        ax.hist(10*np.log10(sn[sn > 1]))
        ax.set_xlabel('Max SNR [dB]')
        ax.set_ylabel('Passes')

        ax = axes[1, ind]
        ax.hist(t[sn > 1])
        ax.set_xlabel('Time past epoch [h]')
        ax.set_ylabel('Passes')

        ax = axes[2, ind]
        ax.plot(t[sn > 1], 10*np.log10(sn[sn > 1]), '.b')
        ax.set_xlabel('Time past epoch [h]')
        ax.set_ylabel('Max SNR [dB]')

    return fig_frags, axes


def animate_fragments_frames(
            t, 
            fragment_states, 
            output_anim, 
            frame_range, 
            cnt_range, 
            plot_radius, 
            max_offset,
        ):
    fig_frag = plt.figure(figsize=(15, 15))
    ax = fig_frag.add_subplot(111, projection='3d')

    size_fixed = False
    offset = 0

    lns = [None for x in fragment_states]
    lnps = [None for x in fragment_states]

    sorts.plotting.grid_earth(ax)
    pbar = tqdm(total=len(frame_range))
    for tid, cnt in zip(frame_range, cnt_range):
        pbar.update(1)
        out_frame = output_anim / f'frag_anim{cnt}.png'
        if out_frame.is_file():
            continue

        if tid <= max_offset:
            offset = tid
        else:
            offset = offset

        for fid, (name, states_fragments) in enumerate(fragment_states.items()):
            if lns[fid] is None:
                lns[fid], = ax.plot(
                    states_fragments[0, (tid - offset):tid], 
                    states_fragments[1, (tid - offset):tid], 
                    states_fragments[2, (tid - offset):tid], 
                    '-b', alpha=0.2,
                )
                lnps[fid], = ax.plot(
                    states_fragments[0, tid:(tid+1)], 
                    states_fragments[1, tid:(tid+1)], 
                    states_fragments[2, tid:(tid+1)], 
                    '.r', alpha=1,
                )
            else:
                lns[fid].set_data(
                    states_fragments[0, (tid - offset):tid], 
                    states_fragments[1, (tid - offset):tid],
                )
                lns[fid].set_3d_properties(states_fragments[2, (tid - offset):tid])
                lnps[fid].set_data(
                    states_fragments[0, tid:(tid+1)], 
                    states_fragments[1, tid:(tid+1)],
                )
                lnps[fid].set_3d_properties(states_fragments[2, tid:(tid+1)])

        if not size_fixed:
            size_fixed = True
        
            ax.set_xlim3d([-plot_radius, plot_radius])
            ax.set_ylim3d([-plot_radius, plot_radius])
            ax.set_zlim3d([-plot_radius, plot_radius])

        fig_frag.savefig(out_frame)
    
    pbar.close()
    plt.close(fig_frag)


# TODO: implement this with multi-core
def animate_fragments(t, fragment_states, output_anim, cores=0):

    plot_radius = []
    for fid, states_fragments in fragment_states.items():
        plot_radius += [np.abs(states_fragments[:3, :]).max()]
    plot_radius = max(plot_radius)

    step_size = 100
    max_offset = 100
    slow_start = 400

    if cores > 1:
        processes = []
        all_frame_range = np.array(list(range(0, slow_start)) + list(range(slow_start, len(t), step_size)))
        for pid in range(cores):
            cnt_range = np.array(range(pid, len(all_frame_range), cores))
            frame_range = all_frame_range[cnt_range]

            processes.append(mp.Process(
                target=animate_fragments_frames, 
                args=(
                    t, 
                    fragment_states, 
                    output_anim, 
                    frame_range, 
                    cnt_range,
                    plot_radius,
                    max_offset,
                ),
            ))
            processes[-1].start()

        for p in processes:
            p.join()

    else:
        frame_range = list(range(0, slow_start)) + list(range(slow_start, len(t), step_size))
        cnt_range = range(len(frame_range))

        animate_fragments_frames(
            t, 
            fragment_states, 
            output_anim, 
            frame_range, 
            cnt_range,
            plot_radius,
            max_offset,
        )
    
    try:
        subprocess.check_call(
            'ffmpeg -start_number 0 -r 6 -i frag_anim%d.png -vcodec mpeg4 ../animation.avi', 
            cwd=str(output_anim.resolve()),
            shell=True,
        )
    except subprocess.CalledProcessError as e:
        logger.info(e)
        logger.info('Could not create movie from animation frames... probably ffmpeg is missing')


def observed_pass_data(radar, data, time_unit='min', obs_epoch=None, figsize=None):

    # plot results
    fig1 = plt.figure(figsize=figsize)
    fig2 = plt.figure(figsize=figsize)
    axes = []
    r_axes = []
    sn_axes = []
    for ind in range(1, len(radar.rx)+1):
        axes.append(fig1.add_subplot(2, len(radar.rx), ind))
        r_axes.append(fig1.add_subplot(2, len(radar.rx), len(radar.rx) + ind))
        sn_axes.append(fig2.add_subplot(1, len(radar.rx), ind))

    tc = {
        's': 1.0,
        'min': 60.0,
        'h': 3600.0,
        'd': 3600.0*24,
    }

    for tx_d in data:
        for rxi, rx_d in enumerate(tx_d):
            for dati, dat in enumerate(rx_d):
                axes[rxi].plot(dat['tx_k'][0, :], dat['tx_k'][1, :], label=f'Pass {dati}')

                if obs_epoch is None:
                    legend = f'Pass {dati}'
                else:
                    start_epoch = obs_epoch + TimeDelta(dat['t'].min(), format='sec')
                    legend = f'Pass {dati} - Start: {start_epoch.iso}'

                sn_axes[rxi].plot(
                    (dat['t'] - np.min(dat['t']))/tc[time_unit], 
                    10*np.log10(dat['snr']),
                    label=legend,
                )
                r_axes[rxi].plot(
                    (dat['t'] - np.min(dat['t']))/tc[time_unit], 
                    (dat['range']*0.5)*1e-3,
                    label=legend,
                )

    axes[0].legend()
    for rxi, ax in enumerate(axes):
        ax.set_xlabel('k_x [East]')
        ax.set_ylabel('k_y [North]')
        ax.set_title(f'Receiver station {rxi}')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
    
    sn_axes[0].legend()
    for rxi, ax in enumerate(sn_axes):
        ax.set_xlabel(f'Time during pass [{time_unit}]')
        ax.set_ylabel('SNR [dB/interval]')
        ax.set_title(f'Receiver station {rxi}')

    r_axes[0].legend()
    for rxi, ax in enumerate(r_axes):
        ax.set_xlabel(f'Time during pass [{time_unit}]')
        ax.set_ylabel('Range [km]')
        ax.set_title(f'Receiver station {rxi}')

    return fig1, fig2
