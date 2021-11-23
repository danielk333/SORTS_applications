import pathlib
import subprocess
import logging

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import TimeDelta

import sorts

logger = logging.getLogger('sst-cli.plotting')


def orbit_sampling(radar, orbit_samples_data, obs_epoch=None, figsize=None):

    txi = 0

    fig_orb = plt.figure(figsize=figsize)
    axes = []
    for rxi in range(1, len(radar.rx)+1):
        axes.append([])
        axes[rxi-1].append(fig_orb.add_subplot(2, len(radar.rx), rxi))
        axes[rxi-1].append(fig_orb.add_subplot(2, len(radar.rx), len(radar.rx)+rxi))

        sn = []
        t = []
        el = []
        for oid in range(len(orbit_samples_data)):
            for ind, d in enumerate(orbit_samples_data[oid][txi][rxi-1]):
                snm = np.argmax(d['snr'])
                sn.append(d['snr'][snm])
                t.append(d['t'][snm])
                el.append(np.degrees(np.arcsin(d['rx_k'][2, snm])))
        t = np.array(t)
        sn = np.array(t)
        el = np.array(el)

        t_sort = np.argsort(t)
        t = t[t_sort]
        sn = sn[t_sort]

        if obs_epoch is None:
            _t = t/3600.0
        else:
            _t = (obs_epoch + TimeDelta(t, format='sec')).datetime

        axes[rxi-1][0].plot(_t, 10*np.log10(sn), '.b')
        axes[rxi-1][0].set_ylabel('Orbit sample SNR [dB]')
        axes[rxi-1][0].set_xlabel('Time past epoch [h]')

        axes[rxi-1][1].plot(_t, el, '.b')
        axes[rxi-1][1].set_ylabel('Elevation [deg]')
        axes[rxi-1][1].set_xlabel('Time past epoch [h]')

    return fig_orb, axes


def fragment_stats(radar, fragment_pass_data, figsize=None):

    fig_frags, axes = plt.subplots(3, len(radar.rx), figsize=figsize)
    for ind in range(0, len(radar.rx)):

        sn = np.array(fragment_pass_data[0][ind]['peak_snr'])
        t = np.array(fragment_pass_data[0][ind]['peak_time'])

        ax = axes[0, ind]
        ax.hist(10*np.log10(sn[sn > 1]))
        ax.set_xlabel('Max SNR [dB]')
        ax.set_ylabel('Passes')

        ax = axes[1, ind]
        ax.hist(t[sn > 1]/3600.0)
        ax.set_xlabel('Time past epoch [h]')
        ax.set_ylabel('Passes')

        ax = axes[2, ind]
        ax.plot(t[sn > 1]/3600.0, 10*np.log10(sn[sn > 1]), '.b')
        ax.set_xlabel('Time past epoch [h]')
        ax.set_ylabel('Max SNR [dB]')

    return fig_frags, axes


def animate_fragments(t, fragment_states, output_anim):

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
            cwd=str(output_anim),
        )
    except subprocess.CalledProcessError as e:
        logger.info(e)
        logger.info('Could not create gif from animation frames... probably ImageMagick is missing')


def observed_pass_data(radar, data, figsize=None):

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

    for tx_d in data:
        for rxi, rx_d in enumerate(tx_d):
            for dati, dat in enumerate(rx_d):
                axes[rxi].plot(dat['tx_k'][0, :], dat['tx_k'][1, :], label=f'Pass {dati}')
                sn_axes[rxi].plot(
                    (dat['t'] - np.min(dat['t']))/(3600.0*24), 
                    10*np.log10(dat['snr']),
                    label=f'Pass {dati}',
                )
                r_axes[rxi].plot(
                    (dat['t'] - np.min(dat['t']))/(3600.0*24), 
                    (dat['range']*0.5)*1e-3,
                    label=f'Pass {dati}',
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
        ax.set_xlabel('Time during pass [d]')
        ax.set_ylabel('SNR [dB/h]')
        ax.set_title(f'Receiver station {rxi}')

    r_axes[0].legend()
    for rxi, ax in enumerate(r_axes):
        ax.set_xlabel('Time during pass [d]')
        ax.set_ylabel('Range [km]')
        ax.set_title(f'Receiver station {rxi}')

    return fig1, fig2
