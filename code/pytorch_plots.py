import numpy as np
import argparse
import os
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import re

model_names = ["MobileNetV2", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]
metrics = ["CPU time total"]

def parse_time(time_str):
    if time_str.endswith("ms"):
        return float(time_str[:-2])
    elif time_str.endswith("us"):
        return float(time_str[:-2]) / 1000.0  # Convert microseconds to milliseconds
    else:
        raise ValueError(f"Unknown time unit in: {time_str}")

def extract_results(results_dir, num_repetitions):
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Error: The input path '{results_dir}' does not exist.")
    
    # Define model types (rows)
    rows = []
    for model in model_names:
        rows.append(model)
        rows.append(model + "_quant")

    # Define columns
    cols = []
    for metric in metrics:
        cols.append(metric)

    # Create an empty DataFrame to store results
    results_df = pd.DataFrame(0.0, index=rows, columns=cols)

    for model in rows:
        for i in range(num_repetitions):
            result_path = os.path.join(results_dir, f"pytorch_{model}_profiling_{i}.txt")
            with open(result_path, 'r') as file:
                output = file.read()
            match = re.search(f"Self CPU time total: (.*?)\n", output, re.DOTALL)
            cpu_time = parse_time(match.group(1).strip())
            results_df.loc[model, "CPU time total"] += cpu_time

    results_df = results_df / num_repetitions 

    return results_df

def plot(results_df, save_dir):
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(f'{save_dir}/pytorch_results.csv')

    titles = {"CPU time total": "Average Inference Time"}

    for metric in metrics:
        means_orig = results_df.loc[model_names, metric].values
        means_quant = results_df.loc[[model + "_quant" for model in model_names], metric].values

        n_groups = len(model_names)
        index = np.arange(n_groups)

        fig, ax = plt.subplots()
        bar_width = 0.35
        opacity = 0.8

        rects1 = plt.bar(index, means_orig, bar_width,
                        alpha=opacity,
                        label='Original')

        rects2 = plt.bar(index + bar_width, means_quant, bar_width,
                        alpha=opacity,
                        label='Quantized')

        plt.xlabel('Model')
        plt.ylabel(f'{titles[metric]} (ms)')
        plt.title(F'PyTorch: {titles[metric]}')
        plt.xticks(index + bar_width / 2, model_names, rotation=45)
        plt.legend()

        plt.tight_layout()

        # Save the plot as an image
        plt.savefig(f'{save_dir}/{metric}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', help='The directory where the PyTorch models\' results are saved', required=True)
    parser.add_argument('--save_dir', help='The directory where the plots should be saved', required = True)
    parser.add_argument('--num_repetitions', type=int, help='The number of profiling files for each model', required = True)
    args = parser.parse_args()

    results_df = extract_results(args.results_dir, args.num_repetitions)
    plot(results_df, args.save_dir)
    print(results_df)