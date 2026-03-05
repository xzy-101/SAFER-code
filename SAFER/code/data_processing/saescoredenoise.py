import random
from datasets import load_from_disk
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to the input dataset")
parser.add_argument("--output_path", type=str, required=True, help="Base path to save the filtered datasets")
args = parser.parse_args()

data_path = args.data_path
output_path = args.output_path

ds = load_from_disk(data_path)
ds = ds["train"] if "train" in ds else ds

ds = ds.sort("saescore", reverse=False)

num_rows = len(ds)
flip_percentages = [0.02, 0.04, 0.06, 0.08, 0.1]

for flip_percentage in flip_percentages:
    flip_count = int(num_rows * flip_percentage)
    print(f"Processing remove_percentage: {flip_percentage*100}%, Remove count: {flip_count}")

    ds_filtered = ds.select(range(flip_count, num_rows))

    save_path = os.path.join(output_path, f"remove_{flip_percentage * 100:.1f}%")
    ds_filtered.save_to_disk(save_path)
    print(f"Filtered dataset saved to {save_path}")
