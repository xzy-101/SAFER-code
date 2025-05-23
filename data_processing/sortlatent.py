import re
latent_path = "../sae/src/safetylatent.txt"
activation_path = "../sae/src/activation_diff.txt"
output_prefix = "../sae/src/top_latents"

latent_values = set()
with open(latent_path, "r") as f:
    for line in f:
        if line.strip() == "":
            continue
        match = re.match(r"^\s*(\d+)", line)
        if match:
            latent_values.add(int(match.group(1)))

activation_data = []
with open(activation_path, "r") as f:
    for line in f:
        if line.strip() == "":
            continue
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        index = int(parts[0])
        value = float(parts[1])
        if index in latent_values:
            activation_data.append((index, value))

activation_data.sort(key=lambda x: abs(x[1]), reverse=True)

limits = [32]

for limit in limits:
    top_n = activation_data[:limit]
    pos_top = [idx for idx, val in top_n if val > 0]
    neg_top = [idx for idx, val in top_n if val < 0]

    pos_output_file = f"{output_prefix}_positive.txt"
    neg_output_file = f"{output_prefix}_negative.txt"

    with open(pos_output_file, "w") as f:
        for idx in pos_top:
            f.write(f"{idx}\n")

    with open(neg_output_file, "w") as f:
        for idx in neg_top:
            f.write(f"{idx}\n")
