# SAFER
## Introduction
This repository contains the implementation of SAFER.

You can download the required dataset and models from the following links:

- [OpenWebText2 Dataset](https://huggingface.co/datasets/segyges/OpenWebText2)
- [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)
- [wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix)
- [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)


## Environment Setup
The code requires two separate environments.

The sae environment uses Python 3.12, and the required packages are listed in requirements_sae.txt. You can install them with:
```bash
pip install -r requirements_sae.txt
```
The RM environment uses Python 3.10.9, and the required packages are listed in requirements_RM.txt. You can install them with:
```bash
pip install -r requirements_RM.txt
```

## Usage
The following directories are already present in the root directory of the project:

```bash
./RM  
./sae  
./data_processing
```
- The 'RM' directory is used for training the reward model.
- The 'sae' directory is used for training the SAE and extracting features to compute score_safe.
- The 'data_processing' directory contains utility functions for processing data during various stages of the pipeline.

We have provided a run.sh script that can be executed to run the code.
