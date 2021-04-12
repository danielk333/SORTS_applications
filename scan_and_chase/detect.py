import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from astropy.time import Time, TimeDelta

import sorts


def get_detections(scheduler, obj, logger, profiler, t_samp = 20.0):

    #see if the object is detected by the scan

    t = np.arange(0.0, scheduler.end_time, t_samp)
    t_offset = (obj.epoch - scheduler.epoch).sec


    logger.info(f'ScanDetections:get_state (t_offset={t_offset})')
    profiler.start('ScanDetections:get_state')

    states = obj.get_state(t - t_offset)

    profiler.stop('ScanDetections:get_state')

    logger.info(f'ScanDetections:find_passes')
    profiler.start('ScanDetections:find_passes')

    all_passes = scheduler.radar.find_passes(t, states, cache_data = True)

    profiler.stop('ScanDetections:find_passes')

    logger.info(f'ScanDetections:observe_passes (pass_num = {len(all_passes[0][0])})')
    profiler.start('ScanDetections:observe_passes')

    grouped_passes = sorts.group_passes(all_passes)

    for passes in grouped_passes[0]:
        #passes is now a list of each rx for the pass itself
        data = scheduler.observe_passes(passes, linear_list = True, space_object = obj, snr_limit=True, epoch = scheduler.epoch)
        data = [x for x in data if x is not None]
        
        _passes = passes
        _data = data

        #we have a detection
        if len(data) > 0:
            break

    if len(grouped_passes[0]) == 0:
        _passes, _data = [], []

    profiler.stop('ScanDetections:observe_passes')

    return t, states, _passes, _data


def orbit_determination(data_select, scheduler, obj, rx_passes, error_cache_path, logger, profiler, Sigma_orb0=None):

    #to simulate a stare and chase, we need to figure out the initial orbit determination errors

    print(f'\nUsing "{error_cache_path}" as cache for LinearizedCoded errors.')
    err = sorts.errors.LinearizedCodedIonospheric(scheduler.radar.tx[0], seed=123, cache_folder=error_cache_path)

    variables = ['x','y','z','vx','vy','vz']
    deltas = [1e-4]*3 + [1e-6]*3 + [1e-2]

    datas = []

    logger.info(f'ScanDetections:calculate_observation_jacobian')
    profiler.start('ScanDetections:calculate_observation_jacobian')

    #observe one pass from all rx stations, including measurement Jacobian
    for rxi in range(len(scheduler.radar.rx)):

        #the Jacobean is stacked as [r_measurements, v_measurements]^T so we stack the measurement covariance equally
        data, J_rx = scheduler.calculate_observation_jacobian(
            rx_passes[rxi], 
            space_object=obj, 
            variables=variables, 
            deltas=deltas, 
            snr_limit=True,
            save_states=True, 
            epoch=scheduler.epoch,
        )
        datas.append(data) #create a rx-list of data

        inds = data_select(data)
        data['data_select'] = inds

        #to account for the stack as [r_measurements, v_measurements]^T
        J_keep = np.full((J_rx.shape[0],), False, dtype=np.bool)
        J_keep[inds] = True
        J_keep[inds + len(data['t'])] = True

        J_rx = J_rx[J_keep,:]

        #now we get the expected standard deviations
        r_stds_tx = err.range_std(data['range'][inds], data['snr'][inds])
        v_stds_tx = err.range_rate_std(data['snr'][inds])

        #Assume uncorrelated errors = diagonal covariance matrix
        Sigma_m_diag_tx = np.r_[r_stds_tx**2, v_stds_tx**2]

        #we simply append the results on top of each other for each station
        if rxi > 0:
            J = np.append(J, J_rx, axis=0)
            Sigma_m_diag = np.append(Sigma_m_diag, Sigma_m_diag_tx, axis=0)
        else:
            J = J_rx
            Sigma_m_diag = Sigma_m_diag_tx

        # print(f'Range errors std [m] (rx={rxi}):')
        # print(r_stds_tx)
        # print(f'Velocity errors std [m/s] (rx={rxi}):')
        # print(v_stds_tx)

    profiler.stop('ScanDetections:calculate_observation_jacobian')

    #diagonal matrix inverse is just element wise inverse of the diagonal
    Sigma_m_inv = np.diag(1.0/Sigma_m_diag)

    #For a thorough derivation of this formula:
    #see Fisher Information Matrix of a MLE with Gaussian errors and a Linearized measurement model
    #without a prior since this is IOD
    if Sigma_orb0 is None:
        Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J)
    else:
        Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J + np.linalg.inv(Sigma_orb0))

    return datas, Sigma_orb


