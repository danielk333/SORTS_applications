#!/usr/bin/env python

'''
NEO detection simulation

#TODO:
Make a commandline interface using the Simulation class and create a interactable object that can be used to input any "object" and get out detection possiblities over the given time frame, can be used as a planning tool

'''
import pathlib
import logging

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time, TimeDelta
from astroquery.jplhorizons import Horizons

import sorts
import pyorb


radar = sorts.radars.eiscat3d_interp

for st in radar.tx + radar.rx:
    st.min_elevation = 0

dt = 3600.0
days = 8.0
LD = 384402e3
t_slice = 3600.0

kernel = pathlib.Path('/home/danielk/IRF/IRF_GITLAB/EPHEMERIS_FILES/de430.bsp')

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

epoch = Time('2029-04-09T00:00:00', format='isot', scale='tdb')

# 99942 Apophis
# SPK ID = 2099942
# MINOR BODY ID = 2004 MN4
jpl_obj = Horizons(
    id='2004 MN4', 
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

obj = sorts.SpaceObject(
    sorts.propagator.Rebound,
    propagator_options = propagator_options,
    state = orb,
    epoch = epoch,
    parameters = dict(
        H = jpl_el['H'].data[0],
        d = H_to_D(jpl_el['H'].data[0], 0.14),
        geometric_albedo = 0.14,
        spin_period = (3600.0*24.0)/1e3,
    ),
)
print(obj)

t = np.arange(0, 3600.0*24.0*days, dt)

states, massive_states = obj.get_state(t)
interpolator = sorts.interpolation.Legendre8(states, t)

from schedulers import TrackingScheduler

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

#plot results
fig = plt.figure(figsize=(15,15))
axes = [
    fig.add_subplot(231),
    fig.add_subplot(232),
    fig.add_subplot(233),
]
sn_axes = [
    fig.add_subplot(234),
    fig.add_subplot(235),
    fig.add_subplot(236),
]

fig = plt.figure(figsize=(15,15))
r_axes = [
    fig.add_subplot(131),
    fig.add_subplot(132),
    fig.add_subplot(133),
]
for tx_d in data:
    for rxi, rx_d in enumerate(tx_d):
        for dati, dat in enumerate(rx_d):
            axes[rxi].plot(dat['tx_k'][1,:], dat['tx_k'][2,:], label=f'Pass {dati}')
            sn_axes[rxi].plot((dat['t'] - np.min(dat['t']))/(3600.0*24), 10*np.log10(dat['snr']), label=f'Pass {dati}')
            r_axes[rxi].plot((dat['t'] - np.min(dat['t']))/(3600.0*24), (dat['range']*0.5)/LD, label=f'Pass {dati}')

axes[0].legend()
for rxi, ax in enumerate(axes):
    ax.set_xlabel('k_x [East]')
    ax.set_ylabel('k_y [North]')
    ax.set_title(f'Receiver station {rxi}')

for rxi, ax in enumerate(sn_axes):
    ax.set_xlabel('Time during pass [h]')
    ax.set_ylabel('SNR [dB]')
    ax.set_title(f'Receiver station {rxi}')

r_axes[0].legend()
for rxi, ax in enumerate(r_axes):
    ax.set_xlabel('Time during pass [h]')
    ax.set_ylabel('Range [LD]')
    ax.set_title(f'Receiver station {rxi}')

#plot results
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[0,:]/LD, states[1,:]/LD, states[2,:]/LD, 'b')
for ctrl in scheduler.controllers:
    for ti in range(len(ctrl.t)):
        ax.plot(
            [radar.tx[0].ecef[0]/LD, ctrl.ecefs[0,ti]/LD], 
            [radar.tx[0].ecef[1]/LD, ctrl.ecefs[1,ti]/LD], 
            [radar.tx[0].ecef[2]/LD, ctrl.ecefs[2,ti]/LD], 
            'g',
        )

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
ax.plot(t/(3600.0*24), np.linalg.norm(states[:3,:], axis=0)/LD)

ax.set_xlabel('Time since epoch [h]')
ax.set_ylabel('Distance from Earth [LD]')

plt.show()