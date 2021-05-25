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

end_hours = 10.0

IOD_width = 5.0
tracker_delay = 5.0

update_interval = 60.0

# pop_trunc = slice(1,2)
pop_trunc = None

class ScanAndFullChase(ScanAndChase):
    def select_followups(self, t, states, **kwargs):
        '''Strategy for followups
        '''
        inds = t > kwargs['start_track']
        #Just track all
        return t[inds], states[:,inds]


track_points = 10

class ScanAndSparseChase(ScanAndChase):
    def select_followups(self, t, states, **kwargs):
        '''Strategy for followups
        '''
        inds = t > kwargs['start_track']

        interpolator = sorts.interpolation.Legendre8(states, t)

        #Select track_points spread out tracklet points
        tn_avalible = (t[inds].max() - t[inds].min())/self.timeslice
        if tn_avalible < track_points:
            tn = int(tn_avalible)
        else:
            tn = track_points

        t_samp = np.linspace(t[inds].min(), t[inds].max(), tn)
        return t_samp, interpolator.get_state(t_samp)


radar = sorts.radars.eiscat3d_interp

pop_cache_path = pathlib.Path('./sample_population.h5')
master_path = pathlib.Path('./celn_100.sim')
error_cache_path = pathlib.Path('/home/danielk/IRF/IRF_GITLAB/SORTS/examples/data')

scan = sorts.scans.Fence(azimuth=90, num=40, dwell=0.1, min_elevation=30)

profiler = sorts.profiling.Profiler()

pop_kw = dict(
    propagator = sorts.propagator.SGP4,
    propagator_options = dict(
        settings = dict(
            in_frame='TEME',
            out_frame='ITRS',
        ),
    ),
)

if pop_cache_path.is_file():
    pop = sorts.Population.load(pop_cache_path, **pop_kw)
else:
    pop = sorts.population.master_catalog(master_path, **pop_kw)
    pop.save(pop_cache_path)

if pop_trunc is not None:
    pop.data = pop.data[pop_trunc]

epoch = pop.get_object(0).epoch

np.random.seed(3487)
profiler.start('total')

#this is for "no track" scenario
def data_select_all(data):
    if len(data['t']) == 0:
        return np.empty((0,), dtype=np.int)
    else:
        return np.argwhere(np.full(data['t'].shape, True, dtype=np.bool)).flatten()



def data_select(data):
    #for this one we assume a time delay of IOD_width seconds after first point, all detections within that scope is used
    #to IOD
    #then another tracker_delay seconds for the tracker to hit the buffer
    if len(data['t']) == 0:
        return np.empty((0,), dtype=np.int)
    else:
        return np.argwhere(data['t'] < data['t'].min() + IOD_width).flatten()


#maybe if we have low lag time, its better to follow up directly than to wait
#while for a slow system it might be better to try to collect all detections from the scan before scheduling updates
#test this out
#
#also test if re-updating orbit should be needed once new tracking data hits the buffer?


class ScanAndChaseSim(sorts.Simulation):
    def __init__(self, pop, sched_cls, *args, **kwargs):
        self.pop = pop
        self.inds = list(range(len(pop)))
        self.sched_cls = sched_cls

        super().__init__(*args, **kwargs)

        self.steps['calc'] = self.calc

    def get_data(self, index, plot=False, no_cache=False):
        ret_ = self.iterative_det_od(index, no_cache=no_cache)

        if ret_ is None:
            print(f'NO DATA FOR OBJECT {index}')
            return ret_

        Sigma_orb__, scan_and_chase_datas, t, states, passes, data, chase_schdeule_time, init_object, true_object, sigmas = ret_

        dat = {}
        dat['index'] = index
        dat['Sigma_orb'] = Sigma_orb__
        dat['scan_and_chase_datas'] = scan_and_chase_datas
        dat['t'] = t
        dat['states'] = states
        dat['passes'] = passes
        dat['data'] = data
        dat['chase_schdeule_time'] = chase_schdeule_time
        dat['init_object'] = init_object
        dat['true_object'] = true_object
        dat['sigmas'] = sigmas

        if plot:
            self.plot(dat)

        return dat

    def print_sigmas(self, dat, plot=False):

        print(f'\nLinear orbit estimator covariance [SI-units]:')

        header = ['','x','y','z','vx','vy','vz']

        for si, sig in enumerate(dat['sigmas']):
            if si == 0:
                print('\nCOV - SCAN ONLY - OD')
            elif si == 1:
                print('\nCOV - SCAN IOD')
            else:
                print(f'\nCOV - CHASE step {si-1}')

            list_sig = sig.tolist()
            list_sig = [[var] + row for row,var in zip(list_sig, header[1:])]
            print(tabulate(list_sig, header, tablefmt="simple", floatfmt="1.3e"))

        for si, sig in enumerate(dat['sigmas']):
            if si == 0:
                print('\nSTD - SCAN ONLY - OD')
            elif si == 1:
                print('\nSTD - SCAN IOD')
            else:
                print(f'\nSTD - CHASE step {si-1}')

            list_sig = [[np.sqrt(sig[ci,ci]) for ci,var in enumerate(header[1:])]]
            print(tabulate(list_sig, header[1:], tablefmt="simple", floatfmt="1.3e"))

        if plot:
            fig, axes = plt.subplots(2,3,figsize=(12,8))
            axes = axes.flatten()

            stds = np.empty((6, len(dat['sigmas'])))
            for si, sig in enumerate(dat['sigmas']):
                for ci in range(6):
                    stds[ci,si] = np.sqrt(sig[ci,ci])
            labels = ['Scan', 'IOD'] + [f'{x}' for x in range(len(dat['sigmas']) - 2)]
            for ci in range(6):
                if ci < 3:
                    unit = 'm'
                else:
                    unit = 'm/s'

                axes[ci].bar(labels, stds[ci,:])
                axes[ci].set_ylabel(f'std({header[ci+1]}) [{unit}]', fontsize=16)
                axes[ci].set_yscale('log')
            fig.suptitle('Covariance matrix diagonal: scan and iterative chase')
            plt.show()


    def plot(self, dat):
        passes = dat['passes']
        data = dat['data']
        scan_and_chase_datas = dat['scan_and_chase_datas']
        index = dat['index']

        obj = self.pop.get_object(index)

        max_snr = 10*np.log10(passes[0].calculate_snr(radar.tx[0], radar.rx[0], obj.d))

        fig = plt.figure(figsize=(12,8))
        axes = np.array([
            [
                fig.add_subplot(221),
                fig.add_subplot(222),
            ],
            [
                fig.add_subplot(223),
                fig.add_subplot(224),
            ],
        ])

        axes[1][1].plot(data[0]['t']/60.0, 10*np.log10(data[0]['snr']), 'xk', alpha=1, label='Scan')

        _, axes = sorts.plotting.observed_parameters([scan_and_chase_datas[0]], passes=[passes[0]], axes=axes)

        axes[1][1].plot(passes[0].t/60.0, max_snr, '-g', alpha=1, label='Optimal')
        axes[1][1].set_ylim([0.0, max_snr.max()])
        L = axes[1][1].legend()
        L.get_texts()[1].set_text('Observed')
        fig.suptitle('Scan and chase')
        fig.savefig(self.get_path(f'chase_index{index}.png'))

        plt.show()


    @sorts.iterable_step(iterable='inds', MPI=False, log=True, reduce=lambda t,x: None)
    def calc(self, index, val, **kwargs):
        return self.iterative_det_od(index, **kwargs)


    def iterative_det_od(self, index, **kwargs):
        no_cache = kwargs.get('no_cache', False)

        if not no_cache:
            data_path = self.get_path(f'calc/{index}_data.pickle').resolve()
            _data = self.load_pickle(data_path)
            if _data is not None:
                return _data

        obj = self.pop.get_object(index)

        sigmas = []

        print('RUNNING CALC')

        print(obj)

        scheduler = self.sched_cls(
            radar = radar, 
            scan = scan, 
            epoch = epoch,
            timeslice = 0.1, 
            end_time = 3600.0*end_hours, 
            logger = self.logger,
            profiler = profiler,
        )
        self.scheduler = scheduler

        t, states, passes, data = detect.get_detections(scheduler, obj, self.logger, profiler, t_samp = 10.0)

        if len(data) == 0:
            return None


        try:
            datas_scan, Sigma_orb_scan = detect.orbit_determination(
                data_select_all, 
                scheduler, 
                obj, 
                passes, 
                error_cache_path, 
                self.logger, 
                profiler,
            )
        except Exception as e:
            self.logger.info('Cannot do IOD')
            self.logger.exception(e)
            return None

        Sigma_orb_scan__ = 0.5*(Sigma_orb_scan + Sigma_orb_scan.T)
        sigmas.append(Sigma_orb_scan__)


        try:
            datas, Sigma_orb = detect.orbit_determination(
                data_select, 
                scheduler, 
                obj, 
                passes, 
                error_cache_path, 
                self.logger, 
                profiler,
            )
        except Exception as e:
            self.logger.info('Cannot do IOD')
            self.logger.exception(e)
            return None

        Sigma_orb__ = 0.5*(Sigma_orb + Sigma_orb.T)

        sigmas.append(Sigma_orb__)

        t_iod = datas[0]['t'][datas[0]['data_select']]

        #first detection is IOD state
        init_epoch = epoch + TimeDelta(t_iod.min(), format='sec')

        #sample IOD covariance
        init_orb = np.random.multivariate_normal(datas[0]['states'][:,0], Sigma_orb__)

        init_orb = sorts.frames.convert(
            init_epoch, 
            init_orb, 
            in_frame='ITRS', 
            out_frame='TEME',
        )
        init_object = sorts.SpaceObject(
            x = init_orb[0],
            y = init_orb[1],
            z = init_orb[2],
            vx = init_orb[3],
            vy = init_orb[4],
            vz = init_orb[5],
            epoch = init_epoch,
            **pop_kw
        )


        true_orb = sorts.frames.convert(
            init_epoch, 
            datas[0]['states'][:,0], 
            in_frame='ITRS', 
            out_frame='TEME',
        )
        true_object = sorts.SpaceObject(
            x = true_orb[0],
            y = true_orb[1],
            z = true_orb[2],
            vx = true_orb[3],
            vy = true_orb[4],
            vz = true_orb[5],
            epoch = init_epoch,
            **pop_kw
        )


        chase_schdeule_time = t_iod.max() + tracker_delay

        scheduler.update(init_object, start_track=chase_schdeule_time)

        if update_interval is not None:
            updates = np.floor((passes[0].end() - passes[0].start())/update_interval)
            for update_num in range(1,int(updates)):

                delta_t = chase_schdeule_time + update_interval*update_num
                scheduler.stop_calc_time = delta_t


                def update_data_select(data):
                    if len(data['t']) == 0:
                        return np.empty((0,), dtype=np.int)
                    else:
                        return np.argwhere(data['t'] < data['t'].min() + delta_t).flatten()

                try:
                    datas, Sigma_orb = detect.orbit_determination(
                        update_data_select, 
                        scheduler, 
                        obj, 
                        passes, 
                        error_cache_path, 
                        self.logger, 
                        profiler,
                        Sigma_orb0 = Sigma_orb__,
                    )
                except Exception as e:
                    self.logger.info('Cannot do IOD')
                    self.logger.exception(e)
                    return None

                Sigma_orb__ = 0.5*(Sigma_orb + Sigma_orb.T)

                sigmas.append(Sigma_orb__)

                #sample OD covariance
                update_orb = np.random.multivariate_normal(datas[0]['states'][:,0], Sigma_orb__)

                update_orb = sorts.frames.convert(
                    init_epoch, 
                    update_orb, 
                    in_frame='ITRS', 
                    out_frame='TEME',
                )
                update_object = sorts.SpaceObject(
                    x = update_orb[0],
                    y = update_orb[1],
                    z = update_orb[2],
                    vx = update_orb[3],
                    vy = update_orb[4],
                    vz = update_orb[5],
                    epoch = init_epoch,
                    **pop_kw
                )

                scheduler.update_tracker(
                    update_object, 
                    delta_t, 
                    start_track=chase_schdeule_time,
                )

        scheduler.stop_calc_time = None
        scan_and_chase_datas = scheduler.observe_passes(
            passes,
            linear_list = True,
            space_object = obj,
            snr_limit = True,
            epoch = scheduler.epoch,
        )

        _data = Sigma_orb__, scan_and_chase_datas, t, states, passes, data, chase_schdeule_time, init_object.state, true_object.state, sigmas
        if not no_cache:
            self.save_pickle(data_path, _data)

        return _data



arg = sys.argv[1].strip().lower()

if len(sys.argv) > 2:
    run = sys.argv[2].strip().lower()
else:
    run = 'run'

if arg == 'full':
    sched_cls = ScanAndFullChase
elif arg == 'sparse':
    sched_cls = ScanAndSparseChase
else:
    raise Exception('No such simulation yet')

sim = ScanAndChaseSim(
    pop = pop,
    sched_cls = sched_cls,
    scheduler = None,
    root = './data',
    profiler = profiler,
)

if (sim.root / arg).is_dir():
    sim.checkout(arg)
else:
    sim.branch(arg, empty=True)

if run == 'norun':
    print(f'sim = {sim}')
    print(f'== COMMANDS ==')
    print('dat = sim.get_data(index, plot=False, no_cache=False)')
    print('sim.plot(dat)')
    print('sim.print_sigmas(dat)')
elif run == 'run':
    sim.run()
else:
    raise Exception('No such command')

profiler.stop('total')
print('\n' + profiler.fmt(normalize='total'))