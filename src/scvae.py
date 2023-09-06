import torch
from torch import nn
from torch.nn import functional as F
import lightning as L

from typing import Dict, Tuple
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from models import Encoder, Decoder, CyclicWeightScheduler

class SplitContrastiveVAE(L.LightningModule):
    '''
    Split contrastive VAE
    # Arguments
        model_name: name of the model
        optimizer_param: dictionary of optimizer parameters
        latent_dim: dimension of the latent space
        base_channel_size: number of channels in the first layer of the encoder and decoder
        num_input_channels: number of input channels
        width: width of the input image
        height: height of the input image
        act_fn: activation function
        n_background_latent: number of latent dimensions for the background
        n_target_latent: number of latent dimensions for the target
        n_technical_latent: number of latent dimensions for the technical background
        encoder_class: encoder class
        decoder_class: decoder class
    '''
    def __init__(
        self,
        model_name: str='scvae',
        optimizer_param: dict={"optimizer": 'Adam', "lr": 1e-3},
        base_channel_size: int=32,
        num_input_channels: int=4,
        width: int=64,
        height: int=64,
        act_fn: object=nn.GELU,
        n_background_latent: int=32, 
        n_target_latent: int=32, 
        n_technical_latent: int=32,
        encoder_class: object=Encoder,
        decoder_class: object=Decoder,
        step_size: float=2000,
        ngene: int=5050,
        wasserstein_penalty: float=1,
        BatchNorm: str=None,
        n_unique_batch: int=34,
        model: str=None,
        batch_latent_dim: int=32,
        scale_factor: float=0.01,
        ):
        super().__init__()
        
        # Model parameters
        self.model_name = model_name
        self.model = model
        self.num_input_channels = num_input_channels
        self.width = width
        self.height = height
        self.base_channel_size = base_channel_size
        self.BatchNorm = BatchNorm
        self.n_unique_batch = n_unique_batch
        self.network_param = {'num_input_channels': num_input_channels, 'variational':True,
                              'BatchNorm': BatchNorm, 'n_unique_batch': n_unique_batch,
                              'base_channel_size':base_channel_size, 'act_fn':act_fn, 
                              'model':model, 'scale_factor': scale_factor}

        # Latent dimensions
        self.n_target_latent = n_target_latent
        self.n_background_latent = n_background_latent
        self.n_technical_latent = n_technical_latent

        # Optimizer parameters
        self.optimizer_param = optimizer_param

        # Loss parameters
        self.wasserstein_penalty = wasserstein_penalty
        
        # Initialize the cyclic weight scheduler
        self.step_size = step_size
        self.kl_weight_scheduler = CyclicWeightScheduler(step_size=self.step_size)

        # Background encoder
        self.z_encoder = encoder_class(latent_dim=self.n_background_latent, **self.network_param)
        # Technical background encoder
        self.y_encoder = encoder_class(latent_dim=self.n_technical_latent, **self.network_param)
        # Salient encoder 
        self.s_encoder = encoder_class(latent_dim=self.n_target_latent, **self.network_param)
        # Decoder from latent variable to distribution parameters in data space.
        self.decoder = decoder_class(
                                    latent_dim=self.n_background_latent + self.n_technical_latent + self.n_target_latent,
                                    batch_latent_dim=batch_latent_dim,
                                    **self.network_param
                                    )
        
        # for the prior adjustment conditional on gene perturbation labels
        self.zprior_embedding = nn.Embedding(ngene, n_background_latent) #ngene is the number of genes, 
        self.sprior_embedding = nn.Embedding(ngene, n_target_latent)

        # for conditioning on batch
        self.batch_embedding = nn.Embedding(self.n_unique_batch, batch_latent_dim)

        # for the gaussian likelihood, the scale parameter is learned
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        # Example input array needed for visualizing the graph of the network
        self.example_input_array = {'background': torch.zeros(2, self.num_input_channels, self.width, self.height),
                                    'target': torch.zeros(2, self.num_input_channels, self.width, self.height),
                                    'background_label': torch.zeros(2, dtype=torch.int32),
                                    'target_label': torch.zeros(2, dtype=torch.int32),
                                    'background_batch': torch.zeros(2, dtype=torch.int32),
                                    'target_batch': torch.zeros(2, dtype=torch.int32),}
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
    
    def get_image_embedding(self, img, batch=None, gene_label=None):
        '''
        img: input image
        batch: batch label for the image
        '''
        qs_m, _ = self.s_encoder(x=img, batch=batch, gene_label=gene_label)
        qz_m, _ = self.z_encoder(x=img, batch=batch, gene_label=gene_label)
        qy_m, _ = self.y_encoder(x=img, batch=batch, gene_label=gene_label)
        return torch.cat((qs_m, qz_m, qy_m), dim=1)

    def forward(self, 
                background, 
                target, 
                background_batch, 
                target_batch, **kwargs):
        '''
        background: background image
        target: target image
        background_batch: batch label for background image
        target_batch: batch label for target image
        '''
        # get embedding for the batch
        inference_outputs = self.inference(background, target)
        # add the batch embeddings
        inference_outputs['background']['batch_embedding'] = self.batch_embedding(background_batch)
        inference_outputs['target']['batch_embedding'] = self.batch_embedding(target_batch)
        # get the decoder outputs
        generative_outputs = self.generative(inference_outputs['background'], 
                                             inference_outputs['target'],)
        recon = {'bg':generative_outputs['background']["px_m"], 
                 "tg":generative_outputs['target']["px_m"]}
        return recon, inference_outputs, generative_outputs
    
    def _generic_inference(self,
                             x: torch.Tensor):
        '''
        x: input image
        '''
        # get the mean and variance of the latent distributions
        qy_m, qy_lv = self.y_encoder(x)
        qz_m, qz_lv = self.z_encoder(x)
        qs_m, qs_lv = self.s_encoder(x)
        
        # sample from latent distribution
        qy_lv = torch.maximum(qy_lv, torch.tensor(-20)) #clipping to prevent going to -inf
        qz_lv = torch.maximum(qz_lv, torch.tensor(-20)) #clipping to prevent going to -inf
        qs_lv = torch.maximum(qs_lv, torch.tensor(-20)) #clipping to prevent going to -inf
        qy_s = torch.exp(qy_lv / 2)
        qz_s = torch.exp(qz_lv / 2)
        qs_s = torch.exp(qs_lv / 2)
        qy = Normal(qy_m, qy_s)
        qz = Normal(qz_m, qz_s)
        qs = Normal(qs_m, qs_s)
        y = qy.rsample()
        z = qz.rsample()
        s = qs.rsample()

        outputs = dict(
            qy_m=qy_m,
            qy_s=qy_s,
            y=y,
            qz_m=qz_m,
            qz_s=qz_s,
            z=z,
            qs_m=qs_m,
            qs_s=qs_s,
            s=s,)
        return outputs
        
    def inference(
            self,
            background: torch.Tensor,
            target: torch.Tensor,
        ) -> Dict[str, Dict[str, torch.Tensor]]:
            background_batch_size = background.shape[0]
            target_batch_size = target.shape[0]
            inference_input = torch.cat([background, target], dim=0)
            outputs = self._generic_inference(inference_input)
            background_outputs, target_outputs = {}, {}
            for key in outputs.keys():
                if outputs[key] is not None:
                    background_tensor, target_tensor = torch.split(
                        outputs[key],
                        [background_batch_size, target_batch_size],
                        dim=0,
                    )
                else:
                    background_tensor, target_tensor = None, None
                background_outputs[key] = background_tensor
                target_outputs[key] = target_tensor
            background_outputs["s"] = torch.zeros_like(background_outputs["s"])
            return dict(background=background_outputs, target=target_outputs)
    
    def _generic_generative(self, 
                              y: torch.Tensor,
                              z: torch.Tensor, 
                              s: torch.Tensor,
                              batch_embedding: torch.Tensor,):
        latent = torch.cat([y, z, s, batch_embedding], dim=-1)
        px_m = self.decoder(latent)
        return dict(px_m=px_m, px_s=self.log_scale)
    
    def generative(
                    self,
                    background: Dict[str, torch.Tensor],
                    target: Dict[str, torch.Tensor]
                    ) -> Dict[str, Dict[str, torch.Tensor]]:
        latent_z_shape = background["z"].shape
        batch_size_dim = 0 if len(latent_z_shape) == 2 else 1
        background_batch_size = background["z"].shape[batch_size_dim]
        target_batch_size = target["z"].shape[batch_size_dim]
        generative_input = {}
        for key in ["y", "z", "s", "batch_embedding"]:
            generative_input[key] = torch.cat(
                [background[key], target[key]], dim=batch_size_dim)
        outputs = self._generic_generative(**generative_input)
        background_outputs, target_outputs = {}, {}
        if outputs["px_m"] is not None:
            background_tensor, target_tensor = torch.split(
                outputs["px_m"],
                [background_batch_size, target_batch_size],
                dim=batch_size_dim,
            )
        else:
            background_tensor, target_tensor = None, None
        background_outputs["px_m"] = background_tensor
        target_outputs["px_m"] = target_tensor
        background_outputs["px_s"] = outputs["px_s"]
        target_outputs["px_s"] = outputs["px_s"]
        return dict(background=background_outputs, target=target_outputs)
    
    def _generic_loss(self, 
                      tensors: torch.Tensor,
                      inference_outputs: Dict[str, torch.Tensor], 
                      generative_outputs: Dict[str, torch.Tensor],
                      prior_mu: Dict[str, torch.Tensor],
                    )-> Dict[str, torch.Tensor]:
        # get the mean and variance of the latent distributions
        qy_m = inference_outputs["qy_m"]
        qy_s = inference_outputs["qy_s"]
        qz_m = inference_outputs["qz_m"]
        qz_s = inference_outputs["qz_s"]
        qs_m = inference_outputs["qs_m"]
        qs_s = inference_outputs["qs_s"]
        px_m = generative_outputs["px_m"]
        px_s = generative_outputs["px_s"]
        # get the prior mean
        sprior_m = prior_mu["sprior_m"]
        zprior_m = prior_mu["zprior_m"]

        # get the reconstruction loss
        recon_loss = self.reconstruction_loss(tensors, px_m, px_s)
        # get the KL divergence
        kl_y = self.latent_kl_divergence(qy_m, qy_s)
        kl_z = self.latent_kl_divergence(qz_m, qz_s, prior_mean=zprior_m)
        kl_s = self.latent_kl_divergence(qs_m, qs_s, prior_mean=sprior_m)
        return dict(recon_loss=recon_loss, kl_y=kl_y, kl_z=kl_z, kl_s=kl_s)
    
    def _get_loss(self, 
             concat_tensors: Dict[str, Tuple[Dict[str, torch.Tensor], int]],
             ):    
        # get the prior mu for the background and target using the gene labels
        background_label = concat_tensors["background_label"].int()
        target_label = concat_tensors["target_label"].int()
        prior_mu = {'background': {'zprior_m': self.zprior_embedding(background_label), 
                                   'sprior_m': self.sprior_embedding(background_label)},
                        "target": {'zprior_m': self.zprior_embedding(target_label), 
                                   'sprior_m': self.sprior_embedding(target_label)}}
        _, inference_outputs, generative_outputs = self.forward(**concat_tensors)            

        # get the loss for the background and target samples
        background_losses = self._generic_loss(
            concat_tensors["background"],
            inference_outputs["background"],
            generative_outputs["background"],
            prior_mu["background"],
        )
        target_losses = self._generic_loss(
            concat_tensors["target"],
            inference_outputs["target"],
            generative_outputs["target"],
            prior_mu["target"],
        )

        # combine the losses
        recon_loss = background_losses["recon_loss"] + target_losses["recon_loss"]
        kl_divergence_y = background_losses["kl_y"] + target_losses["kl_y"]
        kl_divergence_z = + background_losses["kl_z"] + target_losses["kl_z"]
        kl_divergence_s = target_losses["kl_s"]

        wasserstein_loss = (
            torch.norm(inference_outputs["background"]["qs_m"], dim=-1)**2
            + torch.sum(inference_outputs["background"]["qs_s"]**2, dim=-1)
        )

        kl_term_weight = self.kl_weight_scheduler.step()
        elbo = torch.mean(recon_loss + kl_term_weight*(kl_divergence_y + kl_divergence_z + kl_divergence_s + 
                                                       self.wasserstein_penalty*wasserstein_loss))

        # log the losses
        self.log_dict({
            'kl_divergence_y': kl_divergence_y.mean().detach(),
            'kl_divergence_z': kl_divergence_z.mean().detach(),
            'kl_divergence_s': kl_divergence_s.mean().detach(),
            'total_recon_loss': recon_loss.mean().detach(), 
            'wasserstein_loss': wasserstein_loss.mean().detach(),
            'background_recon_loss': background_losses["recon_loss"].mean().detach(),
            'target_recon_loss': target_losses["recon_loss"].mean().detach(),
            'kl_term_weight': kl_term_weight,
        })
        return elbo
    
    @staticmethod
    def reconstruction_loss(sample: torch.Tensor,
                            mean: torch.Tensor, 
                            logscale: torch.Tensor, 
                            ):
        """
        Compute the reconstruction loss for a Gaussian distribution.
        Args:
        ----
            sample: The sample from the Gaussian distribution.
            mean: The mean of the Gaussian distribution.
            logscale: The log of the scale of the Gaussian distribution.
        Returns
        -------
            The reconstruction loss for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        scale = torch.exp(logscale)
        dist = Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return -log_pxz.sum(dim=(1, 2, 3))
    
    @staticmethod
    def latent_kl_divergence(variational_mean, 
                             variational_std, 
                             prior_mean=None,
                             prior_std=None) -> torch.Tensor:
        """
        Compute KL divergence between a variational posterior and standard Gaussian prior.
        Args:
        ----
            variational_mean: Mean of the variational posterior Gaussian.
            variational_var: Variance of the variational posterior Gaussian.
        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        if prior_mean is None:
            prior_mean = torch.zeros_like(variational_mean)
        prior_std = torch.ones_like(variational_std)
        return kl(
            Normal(variational_mean, variational_std),
            Normal(prior_mean, prior_std),
        ).sum(dim=-1)
    
    def configure_optimizers(self):
        lr = self.optimizer_param['lr']
        if self.optimizer_param['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif self.optimizer_param['optimizer'] == 'SGD':
            momentum = self.optimizer_param['momentum']
            nesterov = self.optimizer_param['nesterov']
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "monitor": "train_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("train_loss", loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("val_loss", loss.detach())

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("test_loss", loss.detach())