import numpy as np
from scipy.stats import gamma
import json
import os

def calc_thresholds(losses):

    shape, loc, scale = gamma.fit(losses, floc=0)

    thresholds = {}

    conf_intervals = [0.68, 0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]

    for c in conf_intervals:
        thresholds[str(c)] = gamma.ppf(c, shape, loc=loc, scale=scale)

    as_json = json.dumps(thresholds)

    json_filename = str('./selforacle/losses/thresholds_MNIST.json')

    with open(json_filename, 'w') as fp:
        fp.write(as_json)

    return thresholds
