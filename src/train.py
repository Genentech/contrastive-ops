from pprint import pprint

import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import kornia.augmentation as K
import wandb

from src.constants import Column
from src.transformation import ArcsinhTransform, MinMaxNorm
from src.callbacks import ImagePredictionLogger
from src.helper import get_images, set_seed, get_module
from embed import embed

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="ctvae")
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--step_size", type=int, default=4500)
parser.add_argument("--max_kl_weight", type=float, default=1)
parser.add_argument("--ngene", type=int, default=5050)
parser.add_argument("--n_unique_batch", type=int, default=34)
parser.add_argument("-w2", "--wasserstein_penalty", type=float, default=8)
parser.add_argument("-nz", "--n_z_latent", type=int, default=32)
parser.add_argument("-ns", "--n_s_latent", type=int, default=32)
parser.add_argument("--n_technical_latent", type=int, default=0)
parser.add_argument("--batch_latent_dim", type=int, default=32)
parser.add_argument("--BatchNorm", type=str, default=None)
parser.add_argument("--base_channel_size", type=int, default=32)
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--scale_factor", type=float, default=0.1)
parser.add_argument("--disentangle", action='store_true')
parser.add_argument("--adjust_prior_s", action='store_true', default=True)
parser.add_argument("--adjust_prior_z", action='store_true', default=True)
parser.add_argument("--classify_s", action='store_true')
parser.add_argument("--classify_z", action='store_true')
parser.add_argument("-cw", "--classification_weight", type=float, default=1)
parser.add_argument("--tc_penalty", type=float, default=1)
parser.add_argument("--center_crop", action='store_true')
parser.add_argument("--image_size", type=int, default=64)
parser.add_argument("--reg_type", type=str, default=None)
parser.add_argument("--klscheduler", type=str, default='cyclic')
parser.add_argument("--total_steps", type=int, default=3000)
parser.add_argument("--latent_dim", type=int, default=64)
parser.add_argument("--module", type=str, default='contrastive')

parser.add_argument("--project", type=str, default="ops-training")
parser.add_argument("-l", "--log_model", type=str, default='all')
parser.add_argument("--subname", type=str, default="")
parser.add_argument("--save_dir", type=str, default="/home/wangz222/scratch/")

parser.add_argument("--dataset_path_ntc", type=str, default='/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops/datasets/funk22/funk22_lmdb_shuffled/ntc') # Will be derived from root
parser.add_argument("--dataset_path_perturbed", type=str, default='/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops/datasets/funk22/funk22_lmdb_shuffled/perturbed') # Will be derived from root
parser.add_argument("--plate_list", type=str, nargs='+', default=['20200202_6W-LaC024A', '20200202_6W-LaC024D', '20200202_6W-LaC024E', '20200202_6W-LaC024F', '20200206_6W-LaC025A', '20200206_6W-LaC025B'])
parser.add_argument("--test_ratio", type=float, nargs='+', default=[0.83,0.02,0.15])
parser.add_argument("--save_data_dir", type=str, default="/home/wangz222/scratch/splits_shuffled")
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--batch_correction", action='store_true')
parser.add_argument("--label", type=str, nargs='*', default=[Column.gene.value, Column.batch.value])
parser.add_argument("--stat_path", type=str, default="/home/wangz222/data/per_well_robust_stat.pkl")

parser.add_argument("--max_epochs", type=int, default=2)
parser.add_argument("--gradient_clip_val", type=float, default=0.5)
parser.add_argument("--log_every_n_steps", type=int, default=20)
parser.add_argument("--gradient_clip_algorithm", type=str, default="value")
parser.add_argument("--val_check_interval", type=int, default=0.1)
parser.add_argument("--limit_val_batches", type=int, default=300)
parser.add_argument("--limit_train_batches", type=float, default=1.0)

parser.add_argument("--monitor", type=str, default="val_loss")
parser.add_argument("--min_delta", type=int, default=0)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--mode", type=str, default='min')
args = parser.parse_args()

# Adding code to use the parsed arguments in the dictionaries
model_param = {
                "model_name": args.model_name,
                "optimizer_param": {"optimizer": args.optimizer, "lr": args.lr},
                "step_size": args.step_size,
                "ngene": args.ngene,
                "n_unique_batch": args.n_unique_batch,
                "wasserstein_penalty": args.wasserstein_penalty,
                "n_z_latent": args.n_z_latent,
                "n_s_latent": args.n_s_latent,
                "n_technical_latent": args.n_technical_latent,
                "batch_latent_dim": args.batch_latent_dim,
                "BatchNorm": args.BatchNorm,
                "base_channel_size": args.base_channel_size,
                "model": args.model,
                "scale_factor": args.scale_factor,
                "batch_size": args.batch_size,
                "adjust_prior_s": args.adjust_prior_s,
                "adjust_prior_z": args.adjust_prior_z,
                "classify_s": args.classify_s,
                "classify_z": args.classify_z,
                "classification_weight": args.classification_weight,
                "tc_penalty": args.tc_penalty,
                "disentangle": args.disentangle,
                "image_size": args.image_size,
                "max_kl_weight": args.max_kl_weight,
                "reg_type": args.reg_type,
                "klscheduler": args.klscheduler,
                "total_steps": args.total_steps,
                "latent_dim": args.latent_dim,
                }
logger_p = {
                "project": args.project,
                "name": args.model_name + "_" + args.subname,
                "log_model": args.log_model,
                "save_dir": args.save_dir,
                }
data_param = {
                "dataset_path": {'ntc': args.dataset_path_ntc, 'perturbed': args.dataset_path_perturbed},
                "plate_list": args.plate_list,
                "test_ratio": args.test_ratio,
                "save_dir": args.save_data_dir,
                "loader_param": {"batch_size": args.batch_size, "num_workers": args.num_workers},
                "batch_correction": args.batch_correction,
                "label": [Column.gene.value, Column.batch.value], #put batch at the end if using it
                "stat_path": args.stat_path, #used for batch correction
                }
train_param = {
                "max_epochs": args.max_epochs,
                "gradient_clip_val": args.gradient_clip_val,
                "log_every_n_steps": args.log_every_n_steps,
                "gradient_clip_algorithm": args.gradient_clip_algorithm,
                "val_check_interval": args.val_check_interval,
                "limit_val_batches": args.limit_val_batches,
                "limit_train_batches": args.limit_train_batches,
                }
earlystop = {
                "monitor": args.monitor,
                "min_delta": args.min_delta,
                "patience": args.patience,
                "mode": args.mode,
                }

print(logger_p['name'])
pprint(model_param)
pprint(logger_p)
pprint(data_param)
pprint(train_param)
pprint(earlystop)

def train():
    # set_seed(42)

    # use the appropriate classes
    model_name = model_param['model_name']
    ModelClass = get_module(model_name, 'model')
    DataClass = get_module(args.module, 'dataloader')
    AugmentClass = get_module(args.module, 'augmentation')
    
    # use different transformation depending on if batch correction is on
    if data_param['batch_correction']:
        transform = nn.Sequential(
                          K.Resize(size=(args.image_size, args.image_size), antialias=True), #resize first
                          ArcsinhTransform(factor=1),
                          K.Normalize(mean=(-2.5,-3.1,-1.2,-2.9), std=(9,8,7,8)), # use when performing batch correction    
                          K.Normalize(mean=0.5, std=0.5), # use when performing batch correction 
                         )
    else:
        if args.center_crop:
            print('using center crop')
            transform = nn.Sequential(
                          K.CenterCrop(size=(args.image_size, args.image_size)),
                          ArcsinhTransform(factor=1),
                          K.Normalize(mean=7, std=7), # use when not performing bc
                         )
        else:
            transform = nn.Sequential(
                            K.Resize(size=(args.image_size, args.image_size), antialias=True), #resize first
                            ArcsinhTransform(factor=1),
                            K.Normalize(mean=7, std=7), # use when not performing bc
                            )
    data_param['transform'] = AugmentClass(transform)

    # define datamodule
    dm = DataClass(**data_param)
    print('datamodule defined')

    # get example images to assess reconstruction quality
    dm.prepare_data()
    dm.setup(stage='fit')
    example_img = get_images(8, dm.val_dataloader(), data_param['transform'])

    # wandb login
    wandb.login(host='https://genentech.wandb.io')

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(**logger_p)

    # add your data parameters to the wandb config
    wandb_logger.experiment.config.update(data_param)
    wandb_logger.experiment.log_code(root='/home/wangz222/contrastive-ops/src')

    # define model
    model = ModelClass(**model_param)
    print('model defined')

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        # profiler=profiler,
        callbacks=[
            ModelCheckpoint(monitor='val_loss', mode="min", save_top_k=1, save_weights_only=True, verbose=True),
            ImagePredictionLogger(example_img, every_n_epochs=1),
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=240),
            EarlyStopping(verbose=True, **earlystop),
            ],
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        # deterministic=True,
        # detect_anomaly=True,
        **train_param,
    )
    torch.set_float32_matmul_precision('high')
    print('trainer defined')

    # training model
    trainer.fit(model, dm)  
    torch.cuda.empty_cache()
    print('model trained')
    wandb.finish() 
    
    # embedding
    run_id = wandb_logger.experiment.id
    run_name = wandb_logger.experiment.name
    loader_param = {"batch_size": 4200, "num_workers": data_param['loader_param']['num_workers']}
    embed(run_id=run_id, run_name=run_name, loader_param=loader_param, module=args.module)
    print('embedding computed')    

if __name__ == '__main__':
    train()
    