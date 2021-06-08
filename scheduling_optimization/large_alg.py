
def set_schedule(scheduler, points):
    for i, obj in enumerate(scheduler.pop):
    passes = scheduler.get_passes(obj)
    for ps in passes:
        scheduler.update_tracking(obj, ps, tracklet_points=points[i])


def generate_schedule(scheduler, params, conditions):

    points = np.empty((len(scheduler.pop),), dtype=int)
    points[:] = 2

    set_schedule(scheduler, points)
    Sigma_orbs = scheduler.get_od_errors()

    done = False
    while not done:
        points, done = distribute_points(points, Sigma_orbs, params, **conditions)
        set_schedule(scheduler, points)
        Sigma_orbs = scheduler.get_od_errors()

    return Sigma_orbs, points


def cost_function(Sigma_orbs, priorities):
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
            conditions = {'max_points': 93747},
        )
        return cost_function(Sigma_orbs, priorities)

    x = fminsearch(func, x=0.1)

    return x