import argparse
import os
import shutil
from torch.optim import SGD, Adam
import numpy as np
import torch
from functools import partial
import yaml


from dpipe.config import if_missing, lock_dir, run
from dpipe.io import load
from dpipe.train import train, Policy
from dpipe.train.logging import  WANDBLogger
from dpipe.torch import save_model_state, load_model_state, inference_step
from dpipe.dataset.wrappers import apply, cache_methods
from dpipe.im.metrics import dice_score
from dpipe.predict import add_extract_dims, divisible_shape
from dpipe.train.policy import Schedule, TQDM
from dpipe.torch.functional import weighted_cross_entropy_with_logits
from dpipe.batch_iter import Infinite, load_by_random_id, unpack_args, multiply
from dpipe.im.shape_utils import prepend_dims



from datasets.cc359_ds import CC359
from datasets.msm_ds import MultiSiteMri
from datasets.msm_metric_computer import ComputeMetricsMsm
from unet import UNet2D
from utils.checkpoint_utils import CheckpointsWithBest
from utils.metric_utils import get_pred, sdice, aggregate_metric_probably_with_ids, compute_metrics_probably_with_ids, \
    skip_predict, evaluate_individual_metrics_probably_with_ids_no_pred
from utils.scan_utils import Rescale3D, scale_mri, slicewise, SPATIAL_DIMS, get_random_slice, get_random_patch_2d
from utils.training_utils import train_step, load_model_state_fold_wise
from utils.training_utils import fix_seed
from paths import *


from functools import partial
from torch.optim import *

# Don't remove imports from here:
from utils.training_utils import load_by_gradual_id


class Config:
    def parse(self,raw):
        for k,v in raw.items():
            if type(v) == dict:
                curr_func = v.pop('FUNC')
                return_as_class = v.pop('as_class',False)
                if curr_func not in globals():
                    raise Exception(f'func {curr_func} must be imported')

                for key,val in v.items():
                    if type(val) == str and val in globals():
                        v[key] = globals()[val]
                v = partial(globals()[curr_func],**v)
                if return_as_class:
                    v = v()
            elif v in globals():
                v = globals()[v]
            setattr(self,k,v)
    def __init__(self, raw):
        self._second_round = raw.pop('SECOND_ROUND') if 'SECOND_ROUND' in raw else {}
        self.parse(raw)

    def second_round(self):
        self.parse(self._second_round)

def parse_input():
    cli = argparse.ArgumentParser()
    cli.add_argument("--exp_name", default='debug')
    cli.add_argument("--config")
    cli.add_argument("--device", default='cpu')
    cli.add_argument("--source", default=0,type=int)
    cli.add_argument("--target", default=2,type=int)
    cli.add_argument("--target_size", default=2,type=int)
    cli.add_argument("--batch_size", default=16,type=int)
    cli.add_argument("--seed", default=42,type=int)
    cli.add_argument("--dataset", default='CC359',type=str)

    input_params = cli.parse_args()
    if input_params.dataset not in ['CC359','MSM']:
        raise Exception(f'dataset {input_params.dataset} is not supported')

    if input_params.source == input_params.target:
        if input_params.dataset == 'CC359':
            raise Exception('in ds cc359 source must be different from target')
    else:
        if input_params.dataset == 'MSM':
            raise Exception('in ds msm source must the same as target')
    if input_params.config is None:
        raise Exception('config is required parameter')
    if type(input_params.device) == str and input_params.device.isnumeric():
        input_params.device = int(input_params.device)

    return input_params



if __name__ == '__main__':
    opts = parse_input()


    cfg_path = f"configs/{opts.dataset}/{opts.config}.yml"
    if not os.path.exists(cfg_path):
        raise RuntimeError(f'config path {cfg_path} does not exists')
    cfg = Config(yaml.safe_load(open(cfg_path,'r')))
    pretrain = getattr(cfg,'PRETRAIN',False)
    msm = opts.dataset == 'MSM'
    slice_sampling_interval = 1
    if msm:
        base_res_dir = msm_res_dir
        base_split_dir = msm_splits_dir
    else:
        base_res_dir = cc359_res_dir
        base_split_dir = cc359_splits_dir

    device = opts.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('Warning: using CPU for training might be slow')
    ## define paths
    if pretrain:
        exp_dir = os.path.join(base_res_dir,f'source_{opts.source}',opts.exp_name)
        splits_dir =  os.path.join(base_split_dir,'sources',f'source_{opts.source}')
        opts.target = None
        opts.target_size = None
    else:
        exp_dir = os.path.join(base_res_dir,f'target_size_{opts.target_size}',f'source_{opts.source}_target_{opts.target}',opts.exp_name)
        splits_dir =  os.path.join(base_split_dir,f'ts_{opts.target_size}',f'target_{opts.target}')
    Path(exp_dir).mkdir(parents=True,exist_ok=True)
    log_path = os.path.join(exp_dir,'train_logs')
    saved_model_path = os.path.join(exp_dir,'model.pth')
    saved_model_path_policy = os.path.join(exp_dir,'model_policy.pth')
    test_predictions_path = os.path.join(exp_dir,'test_predictions')
    test_metrics_path = os.path.join(exp_dir,'test_metrics')
    best_test_metrics_path = os.path.join(exp_dir,'best_test_metrics')
    checkpoints_path = os.path.join(exp_dir,'checkpoints')
    data_path = CC359_DATA_PATH
    shutil.copy(cfg_path,os.path.join(exp_dir,'config.yml'))

    train_ids = load(os.path.join(splits_dir,'train_ids.json'))
    if getattr(cfg,'ADD_SOURCE_IDS',False):
        train_ids = load(os.path.join(base_split_dir,'sources',f'source_{opts.source}','train_ids.json')) + train_ids
    val_ids = load(os.path.join(splits_dir,'val_ids.json'))
    test_ids = load(os.path.join(splits_dir,'test_ids.json'))


    ## training params
    n_epochs = cfg.NUM_EPOCHS
    batches_per_epoch = getattr(cfg,'BATCHES_PER_EPOCH',100)
    optimizer_creator = getattr(cfg,'OPTIMIZER',partial(SGD,momentum=0.9, nesterov=True))
    if optimizer_creator.func == SGD:
        base_ckpt_path = os.path.join(base_split_dir,'sources',f'source_{opts.source}','model_sgd.pth')
        optim_state_dict_path = os.path.join(base_split_dir,'sources',f'source_{opts.source}','optimizer_sgd.pth')
    else:
        assert optimizer_creator.func == Adam
        base_ckpt_path = os.path.join(base_split_dir,'sources',f'source_{opts.source}','model_adam.pth')
        optim_state_dict_path = os.path.join(base_split_dir,'sources',f'source_{opts.source}','optimizer_adam.pth')
    batch_size = opts.batch_size
    lr_init = getattr(cfg,'LR_INIT',1e-3)
    if pretrain:
        project = f'wandb_s{opts.source}'
    else:
        project = f'wandb_ts_{opts.target_size}_s{opts.source}_t{opts.target}'
    if msm:
        project = 'msm'+ project[4:]

    if opts.exp_name == 'debug':
        print('debug mode')
        batches_per_epoch = 2
        batch_size = 2
        project = 'debug'
        if len(train_ids) > 2:
            train_ids = train_ids[-4:]
    else:
        lock_dir(exp_dir)

    print(f'running {opts.exp_name}')
    fix_seed(opts.seed)
    if not msm:
        voxel_spacing = (1, 0.95, 0.95)
        preprocessed_dataset = apply(Rescale3D(CC359(data_path), voxel_spacing), load_image=scale_mri)
        dataset = apply(cache_methods(apply(preprocessed_dataset, load_image=np.float16)), load_image=np.float32)
    else:
        dataset = MultiSiteMri(train_ids)



    dice_metric = lambda x, y: dice_score(get_pred(x), get_pred(y))
    sdice_tolerance = 1

    sdice_metric = lambda x, y, i: sdice(get_pred(x), get_pred(y), dataset.load_spacing(i), sdice_tolerance)
    val_metrics = {'dice_score': partial(aggregate_metric_probably_with_ids, metric=dice_metric),
                   'sdice_score': partial(aggregate_metric_probably_with_ids, metric=sdice_metric)}
    n_chans_in = 1
    if msm:
        n_chans_in = 3
    architecture = UNet2D(n_chans_in=n_chans_in, n_chans_out=1, n_filters_init=16)

    architecture.to(device)

    if not pretrain:
        print(f'loading ckpt from {base_ckpt_path}')
        load_model_state_fold_wise(architecture=architecture, baseline_exp_path=base_ckpt_path,modify_state_fn=None)

    logger = WANDBLogger(project=project,dir=exp_dir,entity=None,run_name=opts.exp_name)

    optimizer = optimizer_creator(
        architecture.parameters(),
        lr=lr_init,
        weight_decay=0
    )
    if getattr(cfg,'CONTINUE_OPTIMIZER',False):
        print(f'loading optimizer from path {optim_state_dict_path}')
        optimizer.load_state_dict(torch.load(optim_state_dict_path,map_location=torch.device('cpu')))
        print(optimizer.defaults.items())
        for k,v in optimizer.defaults.items():
            optimizer.param_groups[0][k] = v
        from_step = int(getattr(cfg,'FROM_STEP',0))
        if from_step > 0:
            for param in optimizer.param_groups[0]['params']:
                optimizer.state[param]['step'] = from_step
    lr = Schedule(initial=lr_init, epoch2value_multiplier={45: 0.1, })
    cfg.second_round()
    sample_func = getattr(cfg,'SAMPLE_FUNC',load_by_random_id)
    if 'load_by_gradual_id' in str(type(sample_func)):
        sample_func = partial(sample_func,target_size=opts.target_size)
    criterion = getattr(cfg,'CRITERION',weighted_cross_entropy_with_logits)

    if msm:
        metric_to_use = 'dice'
        msm_metrics_computer = ComputeMetricsMsm(val_ids=val_ids,test_ids=test_ids,logger=logger)
    else:
        metric_to_use = 'sdice_score'




    if not msm:
        @slicewise  # 3D -> 2D iteratively
        @add_extract_dims(2)  # 2D -> (4D -> predict -> 4D) -> 2D
        @divisible_shape(divisor=[8] * 2, padding_values=np.min, axis=SPATIAL_DIMS[1:])
        def predict(image):
            return inference_step(image, architecture=architecture, activation=torch.sigmoid)
        validate_step = partial(compute_metrics_probably_with_ids, predict=predict,
                                load_x=dataset.load_image, load_y=dataset.load_segm, ids=val_ids, metrics=val_metrics)
    else:
        def predict(image):
            return inference_step(image, architecture=architecture, activation=torch.sigmoid)
        msm_metrics_computer.predict = predict
        validate_step = partial(msm_metrics_computer.val_metrices)

    lr_policy = None
    architecture_policy = None
    optimizer_policy = None
    train_kwargs = dict(architecture=architecture, optimizer=optimizer, criterion=criterion,train_step_logger=logger)
    if lr:
        train_kwargs['lr'] = lr

    checkpoints = CheckpointsWithBest(checkpoints_path, {
        **{k: v for k, v in train_kwargs.items() if isinstance(v, Policy)},
        'model.pth': architecture, 'optimizer.pth': optimizer
    },metric_to_use=metric_to_use)
    train_step_func = train_step
    x_patch_size = y_patch_size = np.array([256, 256])
    if msm:
        batch_iter = Infinite(
            sample_func(dataset.load_image, dataset.load_segm, ids=train_ids, random_state=opts.seed),
            unpack_args(get_random_slice, interval=slice_sampling_interval,msm=True),
            multiply(np.float32),
            batch_size=batch_size, batches_per_epoch=batches_per_epoch
        )
    else:
        batch_iter = Infinite(
            sample_func(dataset.load_image, dataset.load_segm, ids=train_ids, random_state=opts.seed),
            unpack_args(get_random_slice, interval=slice_sampling_interval),
            unpack_args(get_random_patch_2d, x_patch_size=x_patch_size, y_patch_size=y_patch_size),
            multiply(prepend_dims),
            multiply(np.float32),
            batch_size=batch_size, batches_per_epoch=batches_per_epoch
        )
    train_model = partial(train,
                          train_step=train_step_func,
                          batch_iter=batch_iter,
                          n_epochs=n_epochs,
                          logger=logger,
                          checkpoints=checkpoints,
                          validate=validate_step,
                          bar=TQDM(),
                          **train_kwargs
                          )
    predict_to_dir = skip_predict
    if msm:
        evaluate_individual_metrics = partial(msm_metrics_computer.test_metrices)
    else:
        final_metrics = {'dice_score': dice_metric, 'sdice_score': sdice_metric}
        evaluate_individual_metrics = partial(
            evaluate_individual_metrics_probably_with_ids_no_pred,
            load_y=dataset.load_segm,
            load_x=dataset.load_image,
            predict=predict,
            metrics=final_metrics,
            test_ids=test_ids,
            logger=logger
        )
    fix_seed(seed=opts.seed)

    run_experiment = run(
        if_missing(lambda p: [train_model(), save_model_state(architecture, p)], saved_model_path),
        load_model_state(architecture, saved_model_path),
        if_missing(predict_to_dir, output_path=test_predictions_path),
        if_missing(evaluate_individual_metrics, results_path=test_metrics_path),
        load_model_state(architecture, checkpoints.best_model_ckpt()),
        if_missing(partial(evaluate_individual_metrics,best='_best'), results_path=best_test_metrics_path),
    )

