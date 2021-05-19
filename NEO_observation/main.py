#!/usr/bin/env python

'''
NEO detection simulation

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


dt = 3600.0
days = 8.0
LD = 384402e3
pass_predict = 32000e3/LD

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
print(prop)

epoch = Time('2029-04-09T00:00:00', format='isot', scale='tdb')
print(epoch)
print(epoch.jd)

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

obj = SpaceOject(
    sorts.propagator.Rebound
    propagator_options = propagator_options,
    state = orb,
    epoch = epoch,
)

init_state = np.squeeze(orb.cartesian)
t = np.arange(0, 3600.0*24.0*days, dt)

states, massive_states = prop.propagate(t, init_state, epoch)

from schedulers import TrackingScheduler

scheduler = TrackingScheduler(radar, t, states)

data = scheduler.observe_passes(
    scheduler.passes, 
    space_object=obj, 
    epoch=None, 
    calculate_snr=True, 
    doppler_spread_integrated_snr=False,
    interpolator=None, 
    snr_limit=True, 
    save_states=False, 
    vectorize=False,
    extended_meta=True,
)

#plot results
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[0,:]/LD, states[1,:]/LD, states[2,:]/LD)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
ax.plot(t/(3600.0*24), np.linalg.norm(states[:3,:], axis=0)/LD)
ax.axhline(pass_predict, color='r')

plt.show()