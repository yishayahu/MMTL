#Supervised Domain Adaptation using Gradients Transfer (MMTL, OCTL)

Pytorch implementation of our methods MMTL, OCTL (will be presented at DART, MICCAI 2022).

## Installation
### install packages
* Install PyTorch from http://pytorch.org
* Install wandb using https://docs.wandb.ai/quickstart

* `git clone https://github.com/deepmind/surface-distance.git`
* `pip install surface-distance/` 
### clone MMTL
* `git clone https://github.com/yishayahu/MMTL.git`
* `cd MMTL`
* ```pip3 install -r requirements.txt```

## Dataset

* Download CC359 from: https://www.ccdataset.com/download
* Download MultiSiteMri (msm) from: https://liuquande.github.io/SAML/
* point paths to the downloaded directories at paths.py


## Train models
### commands:
```
python trainer.py --exp_name {expirement name} --config {config_name} --device {device to use} --source {source site} --target {target site} --dataset {dataset name}
```
* exp_name: name of the current experiment (wil be visible in this name at https://wandb.ai/).
* config: config file to run with
  * 'pretrain_adam' for pretraining using adam optimizer
  * 'finetune_adam' for a regular finetuning using adam optimizer
  * 'mmtl_adam' for finetuning using our method MMTL
  * 'octl_adam' for finetuning using our method OCTL
* dataset:  either CC359 or MSM
* source/target site:
  * in CC359 source and target must be different (train on source and finetune on target).
  * in MSM source and target must be the same (train using all the sites beside the given source).

In example use for pretraining the command:
```
python trainer.py --exp_name pretaining_cc59 --config pretrain_adam --device 0 --source 0 --target 2 --dataset CC359
```
and than run MMTL using the command:
```
python trainer.py --exp_name mmtl_cc59 --config mmtl_adam --device 0 --source 0 --target 2 --dataset CC359
```
### notes:
* the results will be visible at https://wandb.ai/
* source and target can be any number between 0 and 5.

