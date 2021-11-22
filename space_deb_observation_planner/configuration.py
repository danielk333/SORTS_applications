import pathlib
import configparser

HERE = pathlib.Path(__file__).parent.resolve()
CACHE = HERE / '.cache'
MODEL_DIR = CACHE / 'nasa-breakup-model'

DEFAULT_CONFIG = {
    'general': {
        'orekit-data': '',
        'epoch': '',
        'scheduler-file': './.cache/test/scheduler.py',
        't_start': '0',
        't_end': '48.0',
        't_step': '10.0',
        'cores': '6',
    },
    'orbit': {
        'line 1': '1 13552U 82092A   21319.03826954  .00002024  00000-0  69413-4 0  9995',
        'line 2': '2 13552  82.5637 123.6906 0018570 108.1104 252.2161 15.29390138142807',
        'x': '',
        'y': '',
        'z': '',
        'vx': '',
        'vy': '',
        'vz': '',
        'epoch': '',
        'coordinate-system': 'GCRS',
        'd': '',
        'A': '',
        'm': '1000.0',
        'samples': '300',
    },
    'fragmentation': {
        'nbm_param_file': '',
        'nbm_param_default': 'sc',
        'nbm_param_type': 'explosion',
        'fragmentation_time': '10.0',
        'animate': 'no',
    },
}


def get_config(path):

    class CustomConfigParser(configparser.ConfigParser):
        def custom_getfloat(self, *args, **kwargs):
            val = self.get(*args, **kwargs)
            if val:
                return float(val)
            else:
                return None

        def custom_getint(self, *args, **kwargs):
            val = self.get(*args, **kwargs)
            if val:
                return int(val)
            else:
                return None

    config = CustomConfigParser(
        interpolation=None, 
        allow_no_value=True,
    )

    config.read_dict(DEFAULT_CONFIG)
    config.read([path])

    return config
