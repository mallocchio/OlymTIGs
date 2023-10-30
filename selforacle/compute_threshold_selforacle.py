import numpy as np
from scipy.stats import gamma
import json
import os

def calc_and_store_thresholds(losses, model_class: str) -> dict:
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

    json_filename = str(model_class + ".json")

    print("Saving thresholds to %s" % json_filename)

    if os.path.exists(json_filename):
        os.remove(json_filename)

    with open(json_filename, 'a') as fp:
        fp.write(as_json)

    print(thresholds)
    
    return thresholds

def run_threshold(run_folder):
    rec_loss_summary = np.load("compute_rec_losses.npy")
    calc_and_store_thresholds(rec_loss_summary, run_folder)

