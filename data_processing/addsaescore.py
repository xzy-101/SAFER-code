from datasets import load_from_disk, Dataset
import os
import pandas as pd
import argparse

score_file = "../sae/src/SafetyScore.txt"

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to the input dataset")
parser.add_argument("--output_path", type=str, required=True, help="Path to save the output dataset")
args = parser.parse_args()

data_path = args.data_path
output_path = args.output_path

scores = []
with open(score_file, "r") as f:
    for line in f:
        idx, val = line.strip().split()
        idx = int(idx)
        if idx == 0:
            continue
        scores.append(float(val))

ds = load_from_disk(data_path)
ds = ds["train"] if "train" in ds else ds

ds = ds.add_column("saescore", scores)

columns_to_keep = ["chosen", "rejected", "saescore"]
ds = ds.remove_columns([col for col in ds.column_names if col not in columns_to_keep])

ds.save_to_disk(output_path)
