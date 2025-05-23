import random
from datasets import load_from_disk
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to the input dataset")
parser.add_argument("--output_path", type=str, required=True, help="Base path to save the flipped datasets")
args = parser.parse_args()

data_path = args.data_path
output_path = args.output_path

ds = load_from_disk(data_path)
ds = ds["train"] if "train" in ds else ds

ds = ds.sort("saescore", reverse=True)
num_rows = len(ds)

def flip_preferences(example, idx):
    example["chosen"], example["rejected"] = example["rejected"], example["chosen"]
    example["saescore"] = -example["saescore"]
    return example

flip_percentages = [0.005, 0.01, 0.025, 0.05]

for flip_percentage in flip_percentages:
    flip_count = int(num_rows * flip_percentage)
    print(f"Processing flip_percentage: {flip_percentage*100}%, Flip count: {flip_count}")

    def flip_top_samples(example, idx):
        if idx < flip_count:
            return flip_preferences(example, idx)
        return example

    ds_flipped = ds.map(flip_top_samples, with_indices=True)

    save_path = os.path.join(output_path, f"flip_{flip_percentage * 100:.2f}%")
    ds_flipped.save_to_disk(save_path)
    print(f"Modified dataset saved to {save_path}")
