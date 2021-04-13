#!/usr/bin/env python

'''

    #explore the time lag parameter space: figure out at what time lags it stops working
    #- use first detection point vs use max detection point vs use all detection points
    #- pick a few objects (there is a reduced master file we used before)

'''
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from astropy.time import Time, TimeDelta

from scheduler import ScanAndChase
import detect

import sorts

class Obs(sorts.scheduler.StaticList, sorts.scheduler.ObservedParameters):
    pass

radar = sorts.radars.eiscat3d_interp

pop_cache_path = pathlib.Path('./sample_population.h5')
error_cache_path = pathlib.Path('/home/danielk/IRF/IRF_GITLAB/SORTS/examples/data')

profiler = sorts.profiling.Profiler()
logger = sorts.profiling.get_logger('sorts')

pop_kw = dict(
    propagator = sorts.propagator.SGP4,
    propagator_options = dict(
        settings = dict(
            in_frame='TEME',
            out_frame='ITRS',
        ),
    ),
)
pop = sorts.Population.load(pop_cache_path, **pop_kw)
obj = pop.get_object(0)
obj0 = pop.get_object(0)

epoch = pop.get_object(0).epoch

np.random.seed(3487)
profiler.start('total')

obs = Obs(
    radar = radar, 
    controllers = [], 
    logger = logger,
    profiler = profiler,
)
obs.epoch = epoch

t = np.arange(0.0, 24*3600.0, 10.0)
t_offset = (obj.epoch - epoch).sec
states = obj.get_state(t - t_offset)
all_passes = obs.radar.find_passes(t, states, cache_data = True)
passes = sorts.group_passes(all_passes)
ps = passes[0][0][0]

radar.min_SNRdb = 2.0


t_tracks = [
    np.array([0.5*(ps.end() + ps.start())]),
    np.linspace(ps.start()+10, ps.end(), num=3),
    np.linspace(ps.start()+10, ps.end(), num=5),
    np.linspace(ps.start()+10, ps.end(), num=10),
]

d_vec = 10.0**np.linspace(-2,2,num=100)
del obj.parameters['A']

print(obj)

#go to start of pass
obj.out_frame = 'TEME'
dt = ps.start() - t_offset
state0 = obj.get_state(dt)
obj.state.cartesian = state0
obj.state.calculate_kepler()
obj.epoch = obj.epoch + TimeDelta(dt, format='sec')
obj.out_frame = 'ITRS'

print(obj)

Sigma_orbs = np.full((6,6,len(d_vec),len(t_tracks)), np.nan, dtype=np.float64)
snr_peak = np.full((len(d_vec),), np.nan, dtype=np.float64)

cache_sigmas = pathlib.Path('./data/Sigma_orbs.npy')
cache_snrs = pathlib.Path('./data/snr_peak.npy')

if cache_sigmas.is_file():
    Sigma_orbs = np.load(cache_sigmas)
    snr_peak = np.load(cache_snrs)
else:
    for tr_i in range(len(t_tracks)):
        states_track = obj0.get_state(t_tracks[tr_i])
        tracker = sorts.controller.Tracker(
            radar = radar, 
            t = t_tracks[tr_i], 
            t0 = 0.0, 
            t_slice = 0.1,
            ecefs = states_track[:3,:],
            return_copy = True,
        )
        obs.controllers = [tracker]

        for ind, d in enumerate(d_vec):
            obj.parameters['d'] = d

            try:
                datas, Sigma_orb = detect.orbit_determination(
                    lambda data: np.arange(len(data['t']), dtype=np.int), 
                    obs, 
                    obj, 
                    passes[0][0], 
                    error_cache_path,
                    logger = logger,
                    profiler = profiler,
                )
            except Exception as e:
                logger.exception(e)
                continue

            Sigma_orbs[:,:,ind,tr_i] = Sigma_orb
            if tr_i == 0:
                snr_peak[ind] = 10*np.log10(np.max(datas[0]['snr']))

            # fig, axes = sorts.plotting.observed_parameters([datas[0]], passes=[passes[0][0][0]])
            # plt.show()

    np.save(cache_sigmas, Sigma_orbs)
    np.save(cache_snrs, snr_peak)


header = ['','x','y','z','vx','vy','vz']
# for ind in range(len(d_vec)):
#     list_sig = (Sigma_orbs[:,:,ind]).tolist()
#     list_sig = [[var] + row for row,var in zip(list_sig, header[1:])]
#     print(tabulate(list_sig, header, tablefmt="simple"))
#     print('\n'*2)

profiler.stop('total')
print('\n' + profiler.fmt(normalize='total'))

# fig, axes = plt.subplots(6,6,figsize = (12,8))

# for i in range(6):
#     for j in range(6):
#         axes[i,j].semilogx(d_vec, Sigma_orbs[i,j,:])

# fig, axes = plt.subplots(2,3,figsize = (12,8))
# axes = axes.flatten()

# for i in range(6):
#     std = Sigma_orbs[i,i,:].copy()
#     inds = np.logical_not(np.isnan(std))
#     std[inds] = np.log10(np.sqrt(std[inds])*1e-3)
#     axes[i].semilogx(d_vec, std)
#     axes[i].set_ylabel(f'{header[i]}'+' [log10(km)]')
#     axes[i].set_xlabel('Object size [m]')

fig, axes = plt.subplots(2,3,figsize = (12,8))
axes = axes.flatten()

for j in range(len(t_tracks)):
    for i in range(6):
        std = Sigma_orbs[i,i,:,j].copy()
        inds = np.logical_not(np.isnan(std))
        std[inds] = np.sqrt(std[inds])*1e-3
        if i == 0:
            axes[i].semilogy(snr_peak, std, label=f'{len(t_tracks[j])} tracklet points')
            if j==len(t_tracks)-1:
                axes[i].legend()
        else:
            axes[i].semilogy(snr_peak, std)
        axes[i].set_ylabel(f'std({header[i+1]}) [km]')
        axes[i].set_xlabel('Peak SNR [dB]')


plt.show()
