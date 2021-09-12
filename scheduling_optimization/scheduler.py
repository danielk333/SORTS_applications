#!/usr/bin/env python
import pathlib
import numpy as np

import sorts


def get_pass_times(passes):
    '''Takes a set of passes over different RX stations for the same pass and gets time window
    '''
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
    return t_min, t_max, t_pass


class TrackingScheduler(
        sorts.scheduler.StaticList, 
        sorts.scheduler.ObservedParameters,
    ):


    def __init__(self, *args, **kwargs):
        self._od_variables = ['x','y','z','vx','vy','vz']
        self._od_deltas = [1e-4]*3 + [1e-6]*3

        kwargs.setdefault('controllers', [])
        self._time_step = kwargs.pop('time_step', 10.0)
        cache_folder=kwargs.pop('error_model_cache')

        super().__init__(*args, **kwargs)

        self._error_model = sorts.errors.LinearizedCodedIonospheric(
            self.radar.tx[0],
            seed=123, 
            cache_folder=cache_folder,
        )


    def calculate_passes(self):
        for oid, obj in enumerate(self.population):

            if self.epoch is not None and obj.epoch is not None:
                t = self.t_propagate - (obj.epoch - self.epoch).sec
            else:
                t = self.t_propagate

            self.states[oid] = obj.get_state(t)
            self.passes[oid] = radar.find_passes(t, states)

            #ASSUME 1 RX STATION
            self.passes[oid] = sorts.passes.group_passes(self.passes[oid])[0]

            #interpolator in self.epoch time
            self.interpolator[oid] = sorts.interpolation.Legendre8(self.states[oid], self.t_propagate)


    def setup_scheduler(self, population, t_start, t_end, priority, pulses, min_Sigma_orb, epoch=None, t_restriction=None):
        self.population = population
        self.t_start = t_start
        self.t_end = t_end
        self.priority = priority
        self.pulses = pulses
        self.min_Sigma_orb = min_Sigma_orb
        self.t_restriction = t_restriction #2xN mat with restricted times where no points are set
        self.epoch = epoch

        self.states = [None]*len(self.population)
        self.passes = [None]*len(self.population)
        self.interpolator = [None]*len(self.population)

        self.t_propagate = np.arange(self.t_start, self.t_end, self._time_step)

        self.calculate_passes()


    def set_baseline(self):
        '''This function sets the minimum number of samples per object to make the minimum orbit requirements.
        If we move above max pulses, we start dropping objects from end of priority list.
        '''
        pass

    def set_schedule(self, nums):
        '''This is the scheduling algorithm
        '''

        #THE IDEA IS:
        # - start with the baseline
        # - use function to start adding measurement points to objects

        self.controllers = []
        for oid, num in enumerate(nums):
            self.set_track_observations(oid, num)


    def set_track_observations(self, oid, num):

        sub_nums = num//len(self.passes[oid])
        tot_nums = 0

        for pind, passes in enumerate(self.passes[oid]):
            if sub_nums == 0:
                curr_nums = 1
            elif pind == len(self.passes[oid]):
                curr_nums = num - tot_nums
            else:
                curr_nums = sub_nums
            tot_nums += curr_nums

            t_min, t_max, t_pass = get_pass_times(passes)

            if curr_nums > 1:
                t_select = np.linspace(t_min, t_max, num=curr_nums)
            elif curr_nums == 1:
                t_select = np.array([0.5*(t_min + t_max)], dtype=np.float64)
            else:
                continue
            ecefs_select = interpolator.get_state(t_select)

            tracker = sorts.controller.Tracker(
                radar = self.radar, 
                t = t_select, 
                ecefs = ecefs_select[:3,:],
            )

            self.controllers.append(tracker)


    def determine_orbits(self):

        Sigma_orbs = []
        for oid in range(len(self.population)):

            for pind, passes in enumerate(self.passes[oid]):
                t_min, t_max, t_pass = get_pass_times(passes)

                t_od_epoch -= 20.0

                #set object state to right at start of pass
                #This is better for OD
                obj = self.population.get_object(oid)
                obj.propagate(t_od_epoch)

                datas = []

                for rxi in range(len(self.radar.rx)):
                    data, J_rx = scheduler.calculate_observation_jacobian(
                        passes[rxi], 
                        space_object=obj, 
                        variables=variables, 
                        deltas=deltas,
                        snr_limit=False,
                        epoch = self.epoch,
                    )
                    datas.append(data)
                    los_r = data['range'] - datas[0]['range']*0.5
                    r_stds_tx = err.range_std(los_r, data['snr'])
                    v_stds_tx = err.range_rate_std(data['snr'])

                    Sigma_m_diag_tx = np.r_[r_stds_tx**2, v_stds_tx**2]

                    if rxi > 0:
                        J = np.append(J, J_rx, axis=0)
                        Sigma_m_diag = np.append(Sigma_m_diag, Sigma_m_diag_tx, axis=0)
                    else:
                        J = J_rx
                        Sigma_m_diag = Sigma_m_diag_tx

                Sigma_m_inv = np.diag(1.0/Sigma_m_diag)
                
                if pind == 0:
                    Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J)
                else:
                    Sigma_orb0 = Sigma_orb
                    Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J + np.linalg.inv(Sigma_orb0))

            Sigma_orbs.append(Sigma_orb)

        return Sigma_orbs







