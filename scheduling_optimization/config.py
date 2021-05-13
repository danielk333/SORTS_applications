#!/usr/bin/env python
import pathlib
import numpy as np

import sorts

obj = sorts.SpaceObject(
    sorts.propagator.SGP4,
    propagator_options = dict(
        settings = dict(
            in_frame='TEME',
            out_frame='ITRS',
        ),
    ),
    a = 7200e3,
    e = 0.05, 
    i = 75, 
    raan = 79,
    aop = 0,
    mu0 = 0,
    epoch = 53005.0,
    parameters = dict(
        d = 0.1,
    ),
)
epoch = obj.epoch

radar = sorts.radars.eiscat3d

t = np.arange(0,3600*24,5.0)
states = obj.get_state(t)
passes = radar.find_passes(t, states)
passes = sorts.passes.group_passes(passes)[0][0] #pick first pass

for ind, ps in enumerate(passes):
    if ind == 0:
       t_min_min = ps.start()
    else:
        if ps.start() < t_min_min:
            t_min_min = ps.start()

t_min_min -= 20.0

#set object state to right at start of pass
obj.propagate(t_min_min)

t = np.arange(0,20*60.0,5.0)
states = obj.get_state(t)
passes = radar.find_passes(t, states)
passes = sorts.passes.group_passes(passes)[0][0] #pick first pass

for ind, ps in enumerate(passes):
    if ind == 0:
       t_min = ps.start()
       t_max = ps.end()
    else:
        if ps.start() > t_min:
            t_min = ps.start()
        if ps.end() < t_max:
            t_max = ps.end()

t_pass = t_max - t_min

interpolator = sorts.interpolation.Legendre8(states, t)

variables = ['x','y','z','vx','vy','vz']
deltas = [1e-4]*3 + [1e-6]*3

tracker = sorts.controller.Tracker(
    radar = radar, 
    t = None, 
    ecefs = None,
)


err_pth = pathlib.Path('/home/danielk/IRF/IRF_GITLAB/SORTS/examples/data/')

class TrackScheduler(
        sorts.scheduler.StaticList, 
        sorts.scheduler.ObservedParameters,
    ):
    
    def set_observations(self, frac, num):
        '''This is the scheduling algorithm
        '''
        tracker = self.controllers[0]

        dt = t_max - t_min
        t0 = t_min + dt*0.5*(1 - frac)
        t1 = t_max - dt*0.5*(1 - frac)

        t_select = np.linspace(t0, t1, num=num)
        ecefs_select = interpolator.get_state(t_select)

        tracker.ecefs = ecefs_select[:3,:]
        tracker.t = t_select

    def determine_orbit(self):
        err = sorts.errors.LinearizedCodedIonospheric(
            self.radar.tx[0], 
            seed=123, 
            cache_folder=err_pth,
        )

        datas = []

        for rxi in range(len(self.radar.rx)):
            data, J_rx = scheduler.calculate_observation_jacobian(
                passes[rxi], 
                space_object=obj, 
                variables=variables, 
                deltas=deltas,
                snr_limit=False,
            )
            datas.append(data)

            r_stds_tx = err.range_std(data['range'] - datas[0]['range'], data['snr'])
            v_stds_tx = err.range_rate_std(data['snr'])

            Sigma_m_diag_tx = np.r_[r_stds_tx**2, v_stds_tx**2]

            if rxi > 0:
                J = np.append(J, J_rx, axis=0)
                Sigma_m_diag = np.append(Sigma_m_diag, Sigma_m_diag_tx, axis=0)
            else:
                J = J_rx
                Sigma_m_diag = Sigma_m_diag_tx

        Sigma_m_inv = np.diag(1.0/Sigma_m_diag)
        Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J)

        return Sigma_orb, datas




scheduler = TrackScheduler(
    radar = radar, 
    controllers = [tracker],
)
