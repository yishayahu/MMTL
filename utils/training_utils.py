

import random
from typing import Sequence, Callable, Union
import numpy as np
import torch

from dpipe.batch_iter import sample
from dpipe.torch import sequence_to_var,get_device
from dpipe.itertools import pam, squeeze_first
from dpipe.torch import load_model_state
from dpipe.torch.utils import *
from dpipe.torch.model import *


def load_by_gradual_id(*loaders: Callable, ids: Sequence, weights: Sequence[float] = None,
                       random_state: Union[np.random.RandomState, int] = None,batches_per_epoch=100,batch_size=16,ts_size=2,keep_source=True):
    source_ids = ids[:-ts_size]
    target_ids = ids[-ts_size:]
    source_iter = sample(source_ids, weights, random_state)
    target_iter = sample(target_ids, weights, random_state)
    epoch = 0
    while True:

        for _ in range(batches_per_epoch):
            if keep_source:
                from_target = min((epoch//4)+ 1,batch_size-1)
            else:
                from_target = min((epoch//4)+ 1,batch_size)
            from_source = batch_size - from_target
            for _ in range(from_target):
                yield squeeze_first(tuple(pam(loaders, next(target_iter))))
            for _ in range(from_source):
                yield squeeze_first(tuple(pam(loaders, next(source_iter))))
        epoch+=1


def fix_seed(seed=0xBadCafe):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_state_fold_wise(architecture, baseline_exp_path,modify_state_fn=None):
    load_model_state(architecture, path=baseline_exp_path, modify_state_fn=modify_state_fn)



def train_step(*inputs, architecture, criterion, optimizer, n_targets=1, loss_key=None,
               alpha_l2sp=None, reference_architecture=None, train_step_logger=None,use_clustering_curriculum=False,batch_iter_step=None, **optimizer_params):
    architecture.train()
    if n_targets >= 0:
        n_inputs = len(inputs) - n_targets
    else:
        n_inputs = -n_targets

    assert 0 <= n_inputs <= len(inputs)
    inputs = sequence_to_var(*inputs, device=architecture)
    inputs, targets = inputs[:n_inputs], inputs[n_inputs:]
    if alpha_l2sp is not None:
        if reference_architecture is None:
            raise ValueError('`reference_architecture` should be provided for L2-SP regularization.')

        w_diff = torch.tensor(0., requires_grad=True, dtype=torch.float32)
        w_diff.to(get_device(architecture))
        for p1, p2 in zip(architecture.parameters(), reference_architecture.parameters()):
            w_diff = w_diff + torch.sum((p1 - p2) ** 2)
        loss = criterion(architecture(*inputs), *targets) + alpha_l2sp * w_diff
    else:
        loss = criterion(architecture(*inputs), *targets)

    global prev_step
    if train_step_logger is not None and train_step_logger._experiment.step > prev_step and reference_architecture is not None:
        prev_step = train_step_logger._experiment.step




    if loss_key is not None:
        optimizer_step(optimizer, loss[loss_key], **optimizer_params)
        return dmap(to_np, loss)
    if type(loss) == dict:
        optimizer_step(optimizer, loss['total_loss_'], **optimizer_params)
        loss = {k: float(v) for (k, v) in loss.items()}
    else:
        optimizer_step(optimizer, loss, **optimizer_params)
        loss = to_np(loss)
    return loss

