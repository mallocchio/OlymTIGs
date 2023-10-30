import numpy as np
from scipy.stats import gamma
import json
import os

def calc_thresholds(losses, run_folder):
    """
    Calculates all thresholds stores them on a file system
    :param losses: array of shape (n,),
                    where n is the number of training data points, containing the losses calculated for these points
    :param model_class: the identifier of the anomaly detector type
    :return: a dictionary of where key = threshold_identifier and value = threshold_value
    """

    shape, loc, scale = gamma.fit(losses, floc=0)

    thresholds = {}

    conf_intervals = [0.68, 0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]

    for c in conf_intervals:
        thresholds[str(c)] = gamma.ppf(c, shape, loc=loc, scale=scale)

    as_json = json.dumps(thresholds)

    json_filename = str(run_folder + "/thresholds.json")

    print("Saving thresholds to %s" % json_filename)

    with open(json_filename, 'a') as fp:
        fp.write(as_json)

    return thresholds
