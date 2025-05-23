import os
import re
import time
import json
import yaml
import torch
import heapq
import wandb
import random
import tiktoken
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from model import *
from typing import List, Union
from torch.optim import Adam
from openai import AzureOpenAI
from collections import defaultdict
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.hooks import RemovableHandle
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def parse_args():
    parser = argparse.ArgumentParser(description='Configuration on sparse autoencoders pipeline')

    parser.add_argument('--language_model', type=str, required=True, help='Language model name (e.g., "Llama-3.2-1B")')
    parser.add_argument('--model_path', type=str, required=True, help='Language model path')
    parser.add_argument('--hidden_size', type=int, required=True, help='Dimensionality of the input residual stream activation')
    parser.add_argument('--latent_size', type=int, required=True, help='Size of the latent space')
    parser.add_argument('--k', type=int, required=True, help='Hyperparameter k for TopK')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--max_length', type=int, required=True, help='Maximum sequence length')
    parser.add_argument('--device', type=str, required=True, help='Device to run the model on (e.g., "cuda:0", "cpu")')
    parser.add_argument('--use_wandb', type=int, required=True, help='Whether to use wandb for logging')

    parser.add_argument('--data_path', type=str, required=False, help='Path to the dataset')
    parser.add_argument('--wandb_project', type=str, required=False, help='Wandb project name')
    parser.add_argument('--num_epochs', type=int, required=False, help='Number of training epochs')
    parser.add_argument('--lr', type=float, required=False, help='Learning rate for training')
    parser.add_argument('--betas', type=float, nargs=2, required=False, help='Beta values for the optimizer')
    parser.add_argument('--seed', type=int, required=False, help='Random seed for reproducibility')
    parser.add_argument('--layer', type=int, required=False, help='Target layer index, start with 1')
    parser.add_argument('--steps', type=int, required=False, help='Number of step batches for unit norm decoder')

    parser.add_argument('--SAE_path', type=str, required=False, help='Path to the trained SAE model file')
    parser.add_argument('--metric', type=str, required=False, help='Evaluation metric (e.g., "NormMSE", "DeltaCE", "KLDiv")')

    parser.add_argument('--api_base', type=str, required=False, help='OpenAI api base')
    parser.add_argument('--api_key', type=str, required=False, help='OpenAI api key')
    parser.add_argument('--api_version', type=str, required=False, help='OpenAI api version')
    parser.add_argument('--engine', type=str, required=False, help='OpenAI api engine (e.g., "gpt-4o", "gpt-4o-mini")')

    parser.add_argument('--pipe_data_path', type=str, nargs='+', required=False, help='Path to the pipe dataset: train, eval and apply')
    parser.add_argument('--pipe_project', type=str, nargs='+', required=False, help='Wandb project name for pipe: train, eval and pipe')

    args = parser.parse_args()
    return args


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def load_config(config_path: str) -> Config:
    # load config from yaml file
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(config_dict)


class OpenWebTextDataset(Dataset):
    def __init__(self, 
                 folder_path: str, 
                 tokenizer: AutoTokenizer, 
                 max_length: int,
                 keyword: str = 'text'):
        self.folder_path = folder_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.keyword = keyword
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
        self.data = self.load_data()

    def load_data(self):
        """Loads and tokenizes data from files, with Assistant mask processing."""
        data = []
        for file_name in self.file_list:
            file_path = os.path.join(self.folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        words = record.get(self.keyword, '').split()
                        for i in range(0, len(words) - 1, self.max_length):
                            chunk_words = words[i:i + self.max_length]
                            chunk = ' '.join(chunk_words)
                            inputs = self.tokenizer(
                                chunk,
                                return_tensors='pt',
                                max_length=self.max_length,
                                padding='max_length',
                                truncation=True
                            )
                            input_ids = inputs['input_ids'].squeeze(0)
                            attention_mask = inputs['attention_mask'].squeeze(0)
                            data.append((input_ids, attention_mask))
                    except json.JSONDecodeError:
                        print(f'Error decoding JSON in file: {file_path}')
                        continue
            return data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def create_dataloader(
    folder_path: str, 
    tokenizer: AutoTokenizer, 
    batch_size: int, 
    max_length: int,
    keyword: str = 'text'
) -> DataLoader:
    dataset = OpenWebTextDataset(folder_path, tokenizer, max_length, keyword)
    
    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        return input_ids, attention_mask

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)
    return dataloader


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wandb_init(project: str, config: dict, name: str) -> None:
    wandb.init(
        project=project,
        config=config,
        name=name
    )


def get_language_model(model_path: str, device: torch.device) -> tuple:
    '''
    Loads and returns a tokenizer and a language model from the specified model path.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    language_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, return_dict_in_generate=True, output_hidden_states=True
    ).to(device)
    return tokenizer, language_model


def get_outputs(
    cfg, batch: tuple, language_model: nn.Module, device: torch.device
) -> tuple:
    '''
    Extracts model outputs and hidden states from a given batch and language model.
    '''
    input_ids, attention_mask = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        outputs = language_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[cfg.layer]
    return input_ids, attention_mask, outputs, hidden_states


def pre_process(hidden_stats: torch.Tensor, eps: float = 1e-6) -> tuple:
    '''
    :param hidden_stats: Hidden states (shape: [batch, max_length, hidden_size]).
    :param eps: Epsilon value for numerical stability.
    '''
    mean = hidden_stats.mean(dim=-1, keepdim=True)
    std = hidden_stats.std(dim=-1, keepdim=True)
    x = (hidden_stats - mean) / (std + eps)
    return x, mean, std


def Normalized_MSE_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return (((x_hat - x) ** 2).mean(dim=-1) / (x**2).mean(dim=-1)).mean()


@torch.no_grad()
def unit_norm_decoder(model: TopkSAE) -> None:
    '''
    Normalize the decoder weights to unit norm
    '''
    model.decoder.weight.data /= model.decoder.weight.data.norm(dim=0)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f'Saved data to {path}')


def hook_SAE(
    cfg,
    model: TopkSAE,
    hooked_module: nn.Module,
) -> List[RemovableHandle]:

    def hook(module: torch.nn.Module, _, outputs):
        if isinstance(outputs, tuple):
            unpack_outputs = list(outputs)
        else:
            unpack_outputs = [outputs]
        
        x, mu, std = pre_process(unpack_outputs[0])
        latents = model.encode(x)
        x_hat = model.decode(latents)
        unpack_outputs[0] = x_hat * std + mu

        if isinstance(outputs, tuple):
            return tuple(unpack_outputs)
        else:
            return unpack_outputs[0]

    handles = [hooked_module.register_forward_hook(hook)]
    return handles


class LinearWarmupLR(LambdaLR):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, max_lr):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.max_lr = max_lr
        lr_lambda = self.lr_lambda
        super().__init__(optimizer, lr_lambda)

    def lr_lambda(self, step: int):
        if step < self.num_warmup_steps:
            return float(step) / float(max(1, self.num_warmup_steps))
        elif step < self.num_training_steps - self.num_training_steps // 5:
            return 1.0
        else:
            decay_steps = self.num_training_steps - self.num_warmup_steps - self.num_training_steps // 5
            return max(0.0, float(self.num_training_steps - step - self.num_warmup_steps) / float(decay_steps))


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.tokenizer, self.language_model = get_language_model(cfg.model_path, self.device)
        self.dataloader = create_dataloader(cfg.data_path, self.tokenizer, cfg.batch_size, cfg.max_length)
        self.title = f'L{cfg.layer}_K{cfg.k}_{cfg.language_model}_{cfg.data_path.split("/")[-1]}_{cfg.latent_size}'
        self.config_dict = {
            'batch_size': self.cfg.batch_size,
            'num_epochs': self.cfg.num_epochs,
            'lr': self.cfg.lr,
            'steps': self.cfg.steps
        }
        self.model = TopkSAE(cfg.hidden_size, cfg.latent_size, cfg.k)
        self.model.to(self.device)
        self.model.train()
        self.optimizer = Adam(self.model.parameters(), lr=cfg.lr, betas=cfg.betas)
        
        num_training_steps = cfg.num_epochs * len(self.dataloader)
        num_warmup_steps = int(num_training_steps * 0.05)
        self.scheduler = LinearWarmupLR(self.optimizer, num_warmup_steps, num_training_steps, cfg.lr)

    def run(self):
        if self.cfg.use_wandb:
            wandb_init(self.cfg.wandb_project, self.config_dict, self.title)
        curr_loss = 0.0
        unit_norm_decoder(self.model)
        for epoch in range(self.cfg.num_epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                _, _, _, hidden_states = get_outputs(self.cfg, batch, self.language_model, self.device)
                x, _, _ = pre_process(hidden_states)

                _, x_hat = self.model(x)
                loss = Normalized_MSE_loss(x, x_hat)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.scheduler.step()
                curr_loss = loss.item()

                if batch_idx % self.cfg.steps == 0:
                    unit_norm_decoder(self.model)
                if self.cfg.use_wandb:
                    wandb.log({'Normalized_MSE': curr_loss})
                print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {curr_loss}')
        
        if self.cfg.use_wandb:
            wandb.finish()
        unit_norm_decoder(self.model)
        torch.save(self.model.state_dict(), f'../SAE_models/{self.title}.pt')
        return curr_loss


class Evaluater:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.tokenizer, self.language_model = get_language_model(cfg.model_path, self.device)
        self.dataloader = create_dataloader(cfg.data_path, self.tokenizer, cfg.batch_size, cfg.max_length)
        self.title = f'{os.path.splitext(os.path.basename(cfg.SAE_path))[0]}_{os.path.basename(cfg.data_path)}_{cfg.metric}'
        self.config_dict = {
            'batch_size': self.cfg.batch_size,
        }
        self.model = TopkSAE(cfg.hidden_size, cfg.latent_size, cfg.k)
        self.model.load_state_dict(torch.load(cfg.SAE_path, weights_only=True, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        if cfg.metric == 'KLDiv':
            self.hooked_module = self.language_model.get_submodule(f'model.layers.{cfg.layer-1}')
        
        self.num_batches = 0
        self.total_loss = 0.0
        self.total_counts = 0.0

    def KLDiv(
        self, logits_original: torch.Tensor, logits_reconstruct: torch.Tensor
    ) -> torch.Tensor:
        probs_original = F.softmax(logits_original, dim=-1)
        log_probs_reconstruct = F.log_softmax(logits_reconstruct, dim=-1)
        loss = F.kl_div(log_probs_reconstruct, probs_original, reduction='batchmean')
        return loss
    
    @torch.no_grad()
    def run(self):
        if self.cfg.use_wandb:
            wandb_init(self.cfg.wandb_project, self.config_dict, self.title)
        for batch_idx, batch in enumerate(self.dataloader):
            input_ids, attention_mask, outputs, hidden_states= get_outputs(self.cfg, batch, self.language_model, self.device)
            x, _, _ = pre_process(hidden_states)
            _, x_hat = self.model(x)

            if self.cfg.metric == 'NormMSE': 
                loss = Normalized_MSE_loss(x, x_hat)
            
            elif self.cfg.metric == 'KLDiv':
                logits_original = outputs.logits
                handles = hook_SAE(self.cfg, self.model, self.hooked_module)
                logits_reconstruct = self.language_model(input_ids=input_ids, attention_mask=attention_mask).logits
                for handle in handles:
                    handle.remove()
                del input_ids, attention_mask
                torch.cuda.empty_cache()
                loss = self.KLDiv(logits_original, logits_reconstruct)
                
            else:
                raise ValueError(f"Invalid metric: {self.cfg.metric}. Expected one of ['NormMSE', 'DeltaCE', 'KLDiv']")

            self.num_batches += 1
            self.total_loss += loss.item()

            if self.cfg.use_wandb:
                wandb.log({'Batch_loss': loss.item()})
            else:
                print(f'Batch: {batch_idx+1}, Loss: {loss.item()}')
        
        if self.cfg.use_wandb:
            wandb.log({'Avg_loss': self.total_loss / self.num_batches})
            wandb.finish()
        return self.total_loss / self.num_batches

class SAE_pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.title = f'L{cfg.layer}_K{cfg.k}_{cfg.language_model}_{cfg.pipe_data_path[0].split('/')[-1]}_{cfg.latent_size}'
        self.cfg.SAE_path = f'../SAE_models/{self.title}.pt'
        self.result_dict = {}
    
    def train(self):
        set_seed(self.cfg.seed)
        self.cfg.data_path = self.cfg.pipe_data_path[0]
        self.cfg.wandb_project = self.cfg.pipe_project[0]
        trainer = Trainer(self.cfg)
        self.result_dict['Train_Loss'] = trainer.run()
        del trainer
        torch.cuda.empty_cache()
    
    def evaluate(self):
        self.cfg.data_path = self.cfg.pipe_data_path[1]
        self.cfg.wandb_project = self.cfg.pipe_project[1]
        self.cfg.batch_size = self.cfg.batch_size // 2
        for metric in ('NormMSE', 'KLDiv'):
            self.cfg.metric = metric
            evaluater = Evaluater(self.cfg)
            self.result_dict[f'{metric}'] = evaluater.run()
            del evaluater
            torch.cuda.empty_cache()

    def run(self):
        start_time = time.time()

        self.train()
        self.evaluate()

        end_time = time.time()
        self.result_dict['Runtime'] = (end_time - start_time) / 3600

        if self.cfg.use_wandb:
            wandb_init(self.cfg.pipe_project[2], self.result_dict, self.title)
            wandb.finish()
            

