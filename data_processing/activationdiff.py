def read_activation_file(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            index, value = line.strip().split('\t')
            data[int(index)] = float(value)
    return data

def compute_difference(file1, file2):
    data1 = read_activation_file(file1)
    data2 = read_activation_file(file2)

    all_values = list(data1.values()) + list(data2.values())
    mean_value = sum(all_values) / len(all_values) if all_values else 0
    mean_value*=2
    diff = []
    for k in data1.keys():
        if k in data2:
            denominator = data1[k] + data2[k] + mean_value
            diff_value = 0.0
            if denominator != 0:
                diff_value = (data1[k] - data2[k]) / denominator * (1)
            diff.append((k, data1[k], data2[k], diff_value))
    sorted_diff = sorted(diff, key=lambda x: abs(x[3]), reverse=True)
    return sorted_diff


def save_results(results, output_file):
    with open(output_file, 'w') as f:
        for index, value1, value2, diff in results:
            f.write(f"{index}\t{diff}\t{value1}\t{value2}\n")


def save_top_indices(results, output_file):
    positive_indices = [index for index, value1, value2, diff in results if diff > 0]
    negative_indices = [index for index, value1, value2, diff in results if diff < 0]
    indices = [index for index, value1, value2, diff in results]
    top_indices=indices[:256]
    with open(output_file, 'w') as f:
        f.write("\n".join(map(str, top_indices)))


def main():
    file1 = "../sae/src/chosen.txt"
    file2 = "../sae/src/rejected.txt"
    output_file = "../sae/src/activation_diff.txt"
    top_indices_file = "../sae/src/top_latents.txt"
    result = compute_difference(file1, file2)
    save_results(result, output_file)
    save_top_indices(result, top_indices_file)
if __name__ == "__main__":
    main()
