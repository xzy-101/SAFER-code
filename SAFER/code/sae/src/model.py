import torch
import torch.nn as nn
import glob
import os

class TopkSAE(nn.Module):
    '''
    TopK Sparse Autoencoder Implements:
    z = TopK(encoder(x - pre_bias) + latent_bias)
    x_hat = decoder(z) + pre_bias
    '''
    def __init__(
        self, hidden_size: int, latent_size: int, k: int
    ) -> None:
        '''
        :param hidden_size: Dimensionality of the input residual stream activation.
        :param latent_size: Number of latent units.
        :param k: Number of activated latents.
        '''
        assert k <= latent_size, f'k should be less than or equal to {latent_size}'
        super(TopkSAE, self).__init__()
        sae_dir = "code/sae/SAE_models"

        stage2_path = os.path.join(sae_dir, "stage2.pt")
        stage1_path = os.path.join(sae_dir, "stage1.pt")
        if os.path.exists(stage2_path):
            model_path = stage2_path
        elif os.path.exists(stage1_path):
            model_path = stage1_path
        else:
            raise FileNotFoundError(f"Neither stage2.pt nor stage1.pt found in {sae_dir}")
        
        state_dict = torch.load(model_path, map_location="cpu")
        pre_bias = state_dict["pre_bias"]
        latent_bias = state_dict["latent_bias"]
        encoder_weight = state_dict["encoder.weight"]
        decoder_weight = state_dict["decoder.weight"]
        self.pre_bias = nn.Parameter(pre_bias)
        self.latent_bias = nn.Parameter(latent_bias)
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)
        self.encoder.weight = nn.Parameter(encoder_weight)
        self.decoder.weight = nn.Parameter(decoder_weight)

        self.k = k
    
    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.pre_bias
        return self.encoder(x) + self.latent_bias
    
    def get_latents(self, pre_acts: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(pre_acts, self.k, dim=-1)
        latents = torch.zeros_like(pre_acts)
        latents.scatter_(-1, topk.indices, topk.values)
        return latents

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = self.pre_acts(x)
        latents = self.get_latents(pre_acts)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.pre_bias
    
    def forward(self, x: torch.Tensor) -> tuple:
        '''
        :param x: Input residual stream activation (shape: [batch_size, max_length, hidden_size]).
        :return:  latents (shape: [batch_size, max_length, latent_size]).
                  x_hat (shape: [batch_size, max_length, hidden_size]).
        '''
        latents = self.encode(x)
        x_hat = self.decode(latents)
        return latents, x_hat



