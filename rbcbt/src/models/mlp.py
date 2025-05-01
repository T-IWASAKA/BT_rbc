# -*- coding: utf-8 -*-
"""
MLP

models file

@author: iwasaka
"""
import torch
import torch.nn as nn

# こいつは要書き換え
class LinearHead(nn.Module):
    def __init__(
            self, pretrained=None, latent_dim:int=None, num_classes:int=None, num_layers:int=2,
            hidden_head:int=512, dropout_head:float=0.3, frozen:bool=False
            ):
        """
        Parameters
        ----------
        pretrained: pre-trained model

        latent_dim: dimension of the representation

        num_classes: number of classes

        num_layers: number of layers in MLP

        hidden_head: number of hidden units in MLP
            int or list of int

        dropout_head: dropout rate

        """
        super().__init__()
        # check the input
        assert pretrained is not None, "!! pretrained model must be given !!"
        assert latent_dim is not None, "!! latent_dim must be given !!"
        assert num_classes is not None, "!! num_classes must be given !!"
        # pretrained model
        self.pretrained = pretrained
        self.frozen = frozen
        if self.frozen:
            for param in self.pretrained.parameters():
                param.requires_grad = False
        # MLP
        layers = []
        if isinstance(hidden_head, int):
            hidden_head = [hidden_head] * num_layers
        in_features = latent_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_head[i]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_head))
            in_features = hidden_head[i]
        layers.append(nn.Linear(hidden_head[i], num_classes))  # output layer
        self.linear_head = nn.Sequential(*layers)


    def forward(self, x):
        mu, logvar = self.pretrained.encode(x)
        z = self.pretrained.reparameterize(mu, logvar)
        recon = self.pretrained.decode(z)
        logits = self.linear_head(mu)  # use the latent representation for classification
        return logits, recon, mu, logvar


    def vae_loss(self, recon_x, x, mu, logvar, beta=1.0):
        batch_size = x.size(0)
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss
    

    def encode(self, x):
        mu, logvar = self.pretrained.encode(x)
        return mu, logvar

