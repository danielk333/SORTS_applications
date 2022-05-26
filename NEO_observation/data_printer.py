import sys
import pathlib

import pickle
import numpy as np
import matplotlib.pyplot as plt

output_path = pathlib.Path(sys.argv[1])

with open(output_path / 'observation_data.pickle', 'rb') as fh:
    data = pickle.load(fh)

for ps in data:
    ind = np.argmax(ps['snr'])
    print('SNR: ', np.log10(ps['snr'][ind])*10, ' dB')
    print('Range: ', ps['range'][ind]*1e-3/2, ' km')
    print('\n')