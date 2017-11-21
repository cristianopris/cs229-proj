import re

import numpy as np
import os

try:
    import ujson as json # Not necessary for monitor writing, but very useful for monitor loading
except ImportError:
    import json

def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = str(value)
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    value = re.sub('[-\s]+', '-', value)
    return value

def corrcoef(dist_a, dist_b):
    """Returns a scalar between 1.0 and -1.0. 0.0 is no correlation. 1.0 is perfect correlation"""
    dist_a = np.copy(dist_a)  # Prevent np.corrcoef from blowing up on data with 0 variance
    dist_b = np.copy(dist_b)
    dist_a[0] += 1e-12
    dist_b[0] += 1e-12
    return np.corrcoef(dist_a, dist_b)[0, 1]

class JSONLogger(object):
    def __init__(self, fname):
        self.file = open(fname, "w+")

    def writekvs(self, kvs):
        def fix_ndarrays(d):
            for k,v in d.items():
                if hasattr(v, 'dtype'):
                    v = v.tolist()
                    d[k] = v
                if isinstance(v, dict):
                    fix_ndarrays(v)
        fix_ndarrays(kvs)
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()

def model_dir(env_name, experiment_name):
    dir = env_name + '_model' + '/' + experiment_name + '/'
    os.makedirs(dir, exist_ok=True)
    return dir