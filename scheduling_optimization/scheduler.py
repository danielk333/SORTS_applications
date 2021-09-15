#!/usr/bin/env python
import pathlib
import numpy as np
from scipy.stats import multivariate_normal

import sorts
from tqdm import tqdm

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

def get_outer_pass_times(passes):
    '''Takes a set of passes over different RX stations for the same pass and gets time window
    '''
    for ind, ps in enumerate(passes):
        if ind == 0:
           t_min = ps.start()
           t_max = ps.end()
        else:
            if ps.start() < t_min:
                t_min = ps.start()
            if ps.end() > t_max:
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
        cache_folder = kwargs.pop('error_model_cache')

        super().__init__(*args, **kwargs)

        #ASSUME 1 RX STATION
        self._error_model = sorts.errors.LinearizedCodedIonospheric(
            self.radar.tx[0],
            seed=123, 
            cache_folder=cache_folder,
        )


    def setup_scheduler(self, population, t_start, t_end, priority, pulses, min_Sigma_orb, epoch, t_restriction=None):
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


    def calculate_passes(self):
        pbar = tqdm(total=len(self.population), desc='Getting object passes')
        for oid, obj in enumerate(self.population):
            pbar.update(1)

            if self.epoch is not None and obj.epoch is not None:
                t = self.t_propagate - (obj.epoch - self.epoch).sec
            else:
                t = self.t_propagate

            self.states[oid] = obj.get_state(t)
            self.passes[oid] = self.radar.find_passes(t, self.states[oid], cache_data=False)

            #ASSUME 1 RX STATION
            self.passes[oid] = sorts.passes.group_passes(self.passes[oid])[0]

            #interpolator in self.epoch time
            self.interpolator[oid] = sorts.interpolation.Legendre8(self.states[oid], self.t_propagate)
        pbar.close()


    def set_base_observations(self):
        '''This function sets the minimum number of samples per object to make the minimum orbit requirements.
        If we move above max pulses, we start dropping objects from end of priority list.
        '''
        order = np.argsort(self.priority)

        base_nums = np.zeros(self.priority.shape, dtype=np.int32)
        base_nums_dist_ = 0
        for oid in order:
            base_nums[oid] += 1
            base_nums_dist_ += 1
            if base_nums_dist_ >= self.pulses:
                return

        below_baseline = True
        skip = np.full(base_nums.shape, False, dtype=np.bool)
        Sigma_orbs = None
        while below_baseline:
            self.set_schedule(base_nums)
            Sigma_orbs_tmp = Sigma_orbs
            Sigma_orbs = self.determine_orbits(skip = skip)
            for i in range(len(Sigma_orbs)):
                if skip[i]:
                    Sigma_orbs[i] = Sigma_orbs_tmp[i]
            
            skip = np.full(base_nums.shape, True, dtype=np.bool)
            below_baseline = False
            
            for oid, sig in enumerate(Sigma_orbs):
                if np.any(np.sqrt(np.diag(sig)[:3]) > self.min_Sigma_orb[0]) or np.any(np.sqrt(np.diag(sig)[3:]) > self.min_Sigma_orb[1]):
                    below_baseline = True
                    base_nums[oid] += 1
                    skip[oid] = False

                print(np.sqrt(np.diag(sig)[:3]) - self.min_Sigma_orb[0], np.diag(sig)[3:] - self.min_Sigma_orb[1])

        self.base_nums = base_nums
        self.base_pulses = base_nums.sum()


    def spend_pulses(self, pulses, params, x=None):

        if x is None:
            x = self.priority
        
        #print((self.priority + params[1])/params[0], self.priority, params[1], params[0])
        distribution = np.log(1 + (x + params[1])/params[0])

        distribution[np.isinf(distribution)] = 0
        distribution = distribution + np.min(distribution)
        distribution = distribution/np.sum(distribution)

        nums = pulses*distribution
        nums = np.floor(nums)
        ind = np.argmax(x)
        nums[ind] += pulses - np.sum(nums)

        return nums


    def set_schedule(self, nums):
        '''This is the instance scheduling algorithm
        '''

        self.controllers = []
        pbar = tqdm(total=len(self.population), desc='Setting object schedule')
        for oid, num in enumerate(nums):
            pbar.update(1)
            self.set_track_observations(oid, num)
        pbar.close()


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

            ecefs_select = self.interpolator[oid].get_state(t_select)

            tracker = sorts.controller.Tracker(
                radar = self.radar, 
                t = t_select, 
                ecefs = ecefs_select[:3,:],
            )

            self.controllers.append(tracker)

            if tot_nums == num:
                break


    def determine_orbits(self, skip=None):

        Sigma_orbs = []
        pbar = tqdm(total=len(self.population), desc='Determining orbit covariances')
        for oid in range(len(self.population)):
            pbar.update(1)
            if skip is not None:
                if skip[oid]:
                    Sigma_orbs.append(None)
                    continue

            Sigma_orb = None
            obj = self.population.get_object(oid)

            for pind, passes in enumerate(self.passes[oid]):
                
                t_min, t_max, t_pass = get_outer_pass_times(passes)

                t_od_epoch = t_min - 20.0

                #set object state to right at start of pass
                #This is better for OD
                obj.propagate(t_od_epoch)

                if Sigma_orb is None:
                    state__ = multivariate_normal.rvs(mean=obj.state.cartesian.flatten(), cov=Sigma_orb)
                    state__.shape = (6,1)
                    obj.state.cartesian = state__

                datas = []
                J = None

                for rxi in range(len(passes)):
                    data, J_rx = self.calculate_observation_jacobian(
                        passes[rxi], 
                        space_object = obj, 
                        variables = self._od_variables, 
                        deltas = self._od_deltas,
                        snr_limit = True,
                        epoch = self.epoch,
                    )
                    datas.append(data)

                    if data is None:
                        if rxi == 0:
                            break
                        else:
                            continue
                    
                    r_stds_tx = self._error_model.range_std(data['range_rx'], data['snr'])
                    v_stds_tx = self._error_model.range_rate_std(data['snr'])

                    Sigma_m_diag_tx = np.r_[r_stds_tx**2, v_stds_tx**2]

                    if J is not None:
                        J = np.append(J, J_rx, axis=0)
                        Sigma_m_diag = np.append(Sigma_m_diag, Sigma_m_diag_tx, axis=0)
                    else:
                        J = J_rx
                        Sigma_m_diag = Sigma_m_diag_tx

                if J is not None:
                    Sigma_m_inv = np.diag(1.0/Sigma_m_diag)
                    
                    if Sigma_orb is None:
                        Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J)
                    else:
                        Sigma_orb0 = Sigma_orb
                        Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J + np.linalg.inv(Sigma_orb0))

            Sigma_orbs.append(Sigma_orb)

        pbar.close()
        
        return Sigma_orbs







