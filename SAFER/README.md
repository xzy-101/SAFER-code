# SAFER
## Introduction
This repository contains the implementation of SAFER. 

The core logic is located in the `code/` directory, which includes `RM`, `sae`, and `data_processing` modules. The workflow is executed through three sequential stages:
1. **Stage 1 (Initial RM Training):** Run `code/stage1.sh`
2. **Stage 2 (SAE Processing):** Run `code/stage2.sh`
3. **Stage 3 (Poisoning & Retraining):** Run `code/stage3.sh`

## Data and Model Preparation

### 1. Download Dataset
Please download the essential dataset for this project from the link below and place it in the `Dataset` directory. Ensure that the directory contains two subfolders: `RM` and `SAE`.

- Primary Dataset: [Dataset4SAFER](https://huggingface.co/datasets/xxx101/Dataset4SAFER)

*Note: This dataset is derived and processed from sources including OpenWebText2, PKU-SafeRLHF, and wildguardmix.*

### 2. Download Model
Download the base model and place it in the `model` directory. Ensure the folder name is `Llama-3.2-1B-Instruct`.

- Model Link: [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

## Environment Setup
The implementation requires two separate environments to handle different dependencies.

### SAE Environment
- Python version: 3.12
- Installation:
cd SAFER/code
pip install -r requirements_sae.txt

### RM Environment
- Python version: 3.10.9
- Installation:
cd SAFER/code
pip install -r requirements_RM.txt

## Usage
The project structure is organized as follows:

- RM: Used for training the reward model.
- sae: Used for training the Sparse Autoencoder (SAE) and extracting features to compute score_safe.
- data_processing: Utility functions for data transformation during various pipeline stages.

To run the complete pipeline, execute the stage scripts in order:
cd SAFER
conda activate RM
bash code/stage1.sh
conda activate sae
bash code/stage2.sh
conda activate RM
bash code/stage3.sh

Note on Customization: By modifying the parameters in code/stage3.sh, you can adjust the ratio of denoising and poisoning to explore different experimental settings.
Each script handles the specific configuration and environment switching required for that stage of the SAFER framework.