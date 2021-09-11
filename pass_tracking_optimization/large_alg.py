
def distribute_points(points, Sigma_orbs, params, **conditions):
    '''EXAMPLE ????
    '''
    
    points_avalible = points.sum() - conditions['max_points']

    order = conditions['priorities'].argsort()

    points_cache = conditions.get('points_cache', 100)

    #we need to remove points
    if points_avalible < 0:
        order = order[::-1]

        for i in order:
            Sigma = Sigma_orbs[i]:
            if object_lost(Sigma):
                if points_avalible < points_cache
                    points_avalible += points[i]
                    points[i] = 0

        order = order[::-1]

    kept_objs = 0

    for i in order:
        Sigma = Sigma_orbs[i]:
        if object_lost(Sigma):
            if points_avalible > 0:
                points_avalible -= 1
                points[i] += 1
        else:
            kept_objs += 1

    done = kept_objs > params['x']*len(Sigma_orbs)

    return points, done

def set_schedule(scheduler, points):
    cnt = 0
    for i, obj in enumerate(scheduler.pop):
        passes = scheduler.get_passes(obj)
        scheduler.update_tracking(obj, passes, tracklet_points=points[cnt])



def generate_schedule(scheduler, params, conditions):

    points = np.empty((len(scheduler.trackers),), dtype=int)
    points[:] = 2

    set_schedule(scheduler, points)
    Sigma_orbs = scheduler.get_od_errors()

    done = False
    iters = 0
    while not done:
        points, done = distribute_points(points, Sigma_orbs, params, **conditions)
        set_schedule(scheduler, points)
        Sigma_orbs = scheduler.get_od_errors()
        iters += 1
        if iters > 10:
            break

    return Sigma_orbs, points


def cost_function(Sigma_orbs, points, priorities):
    c = 0
    tracked = 0
    for Sigma, prio in zip(Sigma_orbs, priorities):
        if Sigma is not None:
            c += error_cost(Sigma)*prio
            tracked += 1

    c *= number_efficiency_cost(tracked)

    return c


def find_schedule(scheduler, priorities):

    def func(x):
        Sigma_orbs, points = generate_schedule(
            scheduler, 
            params = {'x': x},
            conditions = {
                'max_points': 93747,
                'priorities': priorities,
            },
        )
        return cost_function(Sigma_orbs, points, priorities)

    x = fminsearch(func, x=0.1)

    return x