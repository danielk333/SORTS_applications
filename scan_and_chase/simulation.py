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

update_interval = 120.0

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
logger = sorts.profiling.get_logger('StareAndChase')

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
        self.objs = [obj for obj in pop]
        self.sched_cls = sched_cls

        super().__init__(*args, **kwargs)

        self.steps['calc'] = self.calc

    def plot(self, index, obj, *args, **kwargs):
        data_path = self.get_path(f'calc/{index}_data.pickle').resolve()
        _data = self.load_pickle(data_path)
        if _data is not None:
            Sigma_orb__, scan_and_chase_datas, t, states, passes, data, chase_schdeule_time, init_object, true_object = _data
        else:
            return

        print('RUNNING PLOT')

        print(obj)

        print(f'\nLinear orbit estimator covariance [SI-units] (shape={Sigma_orb__.shape}):')

        header = ['','x','y','z','vx','vy','vz']

        list_sig = (Sigma_orb__).tolist()
        list_sig = [[var] + row for row,var in zip(list_sig, header[1:])]

        print(tabulate(list_sig, header, tablefmt="simple"))

        print('='*20)
        print('Init object')
        print(init_object)
        print('='*20 + '\n')

        print('='*20)
        print('True object')
        print(true_object)
        print('='*20)

        figsize = (12,8)

        max_snr = 10*np.log10(passes[0].calculate_snr(radar.tx[0], radar.rx[0], obj.d))

        print(f'Time for scheduler to inject chase: {chase_schdeule_time} sec')

        fig, axes = sorts.plotting.observed_parameters([data[0]], passes=[passes[0]], figsize = figsize)

        axes[1][1].plot(passes[0].t/60.0, max_snr, '-g', alpha=1)
        axes[1][1].set_ylim([0.0, max_snr.max()])
        fig.suptitle('Scan')
        fig.savefig(self.get_path(f'scan_index{index}.png'))
        plt.close(fig)

        fig, axes = sorts.plotting.observed_parameters([scan_and_chase_datas[0]], passes=[passes[0]], figsize = figsize)

        axes[1][1].plot(passes[0].t/60.0, max_snr, '-g', alpha=1)
        axes[1][1].set_ylim([0.0, max_snr.max()])
        fig.suptitle('Scan and chase')
        fig.savefig(self.get_path(f'chase_index{index}.png'))
        plt.close(fig)

        plt.show()



    @sorts.MPI_action(action='barrier')
    @sorts.iterable_step(iterable='objs', MPI=False, log=True, reduce=lambda t,x: None)
    @sorts.pre_post_actions(post='plot')
    @sorts.cached_step(caches='pickle')
    def calc(self, index, obj, **kwargs):

        print('RUNNING CALC')

        print(obj)

        scheduler = self.sched_cls(
            radar = radar, 
            scan = scan, 
            epoch = epoch,
            timeslice = 0.1, 
            end_time = 3600.0*end_hours, 
            logger = logger,
            profiler = profiler,
        )

        t, states, passes, data = detect.get_detections(scheduler, obj, logger, profiler, t_samp = 10.0)

        if len(data) == 0:
            return None

        try:
            datas, Sigma_orb = detect.orbit_determination(
                data_select, 
                scheduler, 
                obj, 
                passes, 
                error_cache_path, 
                logger, 
                profiler,
            )
        except Exception as e:
            self.logger.info('Cannot do IOD')
            self.logger.exception(e)
            return None

        Sigma_orb__ = 0.5*(Sigma_orb + Sigma_orb.T)

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
                        logger, 
                        profiler,
                        Sigma_orb0 = Sigma_orb__,
                    )
                except Exception as e:
                    self.logger.info('Cannot do IOD')
                    self.logger.exception(e)
                    return None

                Sigma_orb__ = 0.5*(Sigma_orb + Sigma_orb.T)

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

        return Sigma_orb__, scan_and_chase_datas, t, states, passes, data, chase_schdeule_time, init_object.state, true_object.state



arg = sys.argv[1].strip().lower()

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

sim.run()

profiler.stop('total')
print('\n' + profiler.fmt(normalize='total'))