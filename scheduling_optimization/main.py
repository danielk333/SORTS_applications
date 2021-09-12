#!/usr/bin/env python
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm

import sorts

#local
import scheduler



if __name__=='__main__':
    scheduler = TrackScheduler(
        radar = radar,
        population = pop, 
    )

    #test_scheduling_set()
    radar = sorts.radars.eiscat3d
    err_pth = pathlib.Path('/home/danielk/IRF/IRF_GITLAB/SORTS/examples/data/')


