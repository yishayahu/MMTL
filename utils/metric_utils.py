import os
from collections import defaultdict
from typing import Sequence, Callable
import surface_distance.metrics as surf_dc
import numpy as np
from scipy import ndimage
from medpy import metric
from tqdm import tqdm

from dpipe.itertools import zip_equal
from dpipe.io import save_json

def sdice(a, b, spacing, tolerance):
    surface_distances = surf_dc.compute_surface_distances(a, b, spacing)
    return surf_dc.compute_surface_dice_at_tolerance(surface_distances, tolerance)

def get_pred(x, threshold=0.5):
    return x > threshold

def aggregate_metric_probably_with_ids(xs, ys, ids, metric, aggregate_fn=np.mean):
    """Aggregate a `metric` computed on pairs from `xs` and `ys`"""
    try:
        return aggregate_fn([metric(x, y, i) for x, y, i in zip_equal(xs, ys, ids)])
    except TypeError:
        return aggregate_fn([metric(x, y) for x, y in zip_equal(xs, ys)])

def evaluate_with_ids(y_true: Sequence, y_pred: Sequence, ids: Sequence[str], metrics: dict) -> dict:
    return {name: metric(y_true, y_pred, ids) for name, metric in metrics.items()}

def compute_metrics_probably_with_ids(predict: Callable, load_x: Callable, load_y: Callable, ids: Sequence[str],
                                      metrics: dict):
    return evaluate_with_ids(list(map(load_y, ids)), [predict(load_x(i)) for i in ids], ids, metrics)


def eval_dice(gt_y, pred_y):
    dice = []
    for cls in range(1,2):
        gt = np.zeros(gt_y.shape)
        pred = np.zeros(pred_y.shape)#np.zeros(pred_y.shape)
        gt[gt_y == cls] = 1
        pred[pred_y == cls] = 1
        dice_this = (2*np.sum(gt*pred))/(np.sum(gt)+np.sum(pred))
        dice.append(dice_this)

    return dice

def connectivity_region_analysis(mask):

    label_im, nb_labels = ndimage.label(mask)#, structure=s)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return label_im

def eval_average_surface_distances(reference, result):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    return metric.binary.asd(result, reference)


def skip_predict(output_path):
    print(f'>>> Passing the step of saving predictions into `{output_path}`', flush=True)
    os.makedirs(output_path)


def evaluate_individual_metrics_probably_with_ids_no_pred(load_y, load_x, predict, metrics: dict, test_ids,
                                                          results_path, exist_ok=False,logger=None,best=''):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for _id in tqdm(test_ids):
        target = load_y(_id)
        prediction = predict(load_x(_id))

        for metric_name, metric in metrics.items():
            try:
                results[metric_name][_id] = metric(target, prediction, _id)
            except TypeError:
                results[metric_name][_id] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)
        if logger is not None:
            logger.value(f'test_{metric_name}{best}',np.mean(list(result.values())))
