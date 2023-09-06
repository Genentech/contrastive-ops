import wandb
import torch
import torchvision
from lightning.pytorch.callbacks import Callback

class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, every_n_epochs=1):
        super().__init__()
        self.val_imgs = val_samples  # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            if type(self.val_imgs) is dict:
                val_imgs = {key: val.to(pl_module.device) for key, val in self.val_imgs.items()}
            elif type(self.val_imgs) is tuple:
                val_imgs = (self.val_imgs[0].to(pl_module.device), self.val_imgs[1].to(pl_module.device))
            else:
                val_imgs = self.val_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                if type(val_imgs) is dict:
                    reconst_imgs = pl_module(**val_imgs)
                elif isinstance(val_imgs, tuple) or isinstance(val_imgs, list):
                    reconst_imgs = pl_module(*val_imgs)
                else:
                    reconst_imgs = pl_module(val_imgs)
                if isinstance(reconst_imgs, tuple):
                    reconst_imgs = reconst_imgs[0]
                if isinstance(val_imgs, tuple):
                    val_imgs = val_imgs[0]
                if isinstance(val_imgs, dict):
                    val_imgs = {"background":val_imgs['background'], "target":val_imgs['target']}
                pl_module.train()
            # Plot and add to tensorboard
            reconst_imgs = torch.cat(list(reconst_imgs.values()), dim=0) if type(reconst_imgs) is dict else reconst_imgs
            val_imgs = torch.cat(list(val_imgs.values()), dim=0) if type(val_imgs) is dict else val_imgs
            imgs = torch.stack([val_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=8, normalize=True, range=(-1, 1))
            trainer.logger.experiment.log({"Reconstructions":
                                           wandb.Image(grid, caption="Left: Input, Right: Output"),
                                           "global_steps": trainer.global_step})

class EarlyStoppingOnTrainBatchEnd(Callback):   
    '''
    Stop training if loss hasn't improved in a while, 
    '''
    def __init__(self, monitor, min_delta, patience, check_freq):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.check_freq = check_freq
        self.losses = []
        self.no_improve_counter = 0
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.check_freq == 0:
            loss = trainer.callback_metrics.get(self.monitor)
            if loss is not None:
                if len(self.losses) > 0:
                    if abs(self.losses[-1] - loss) < self.min_delta:
                        self.no_improve_counter += 1
                        if self.no_improve_counter >= self.patience:
                            trainer.should_stop = True
                            print("Stopping early because loss hasn't improved in {} checks".format(self.patience))
                    else:
                        self.no_improve_counter = 0  # reset counter

                self.losses.append(loss)