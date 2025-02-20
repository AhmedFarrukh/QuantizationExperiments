import argparse
import pathlib
import os
import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

model_names = ["MobileNetV2", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]
metrics = ["Init Time", "Init Inference", "First Inference", "Warmup Inference", "Avg Inference", "Memory Init", "Memory Overall"]

def parse_benchmark_output(output, results, num_repetitions):
    """
    Parse benchmark output to extract model initialization times, inference timings, and memory footprint.
    """
    # Regular expressions to match the required information
    init_time_pattern = re.compile(r'INFO: Initialized session in (.+?)ms.')
    inference_pattern = re.compile(r'INFO: Inference timings in us: Init: (.+?), First inference: (.+?), Warmup \(avg\): (.+?), Inference \(avg\): (.+?)$')
    memory_pattern = re.compile(r'INFO: Memory footprint delta from the start of the tool \(MB\): init=(.+?) overall=(.+?)$')
    inference_std_pattern = re.compile(r'std=(.+?) ')

    for line in output.split('\n'):
        # Match the initialization time
        init_match = init_time_pattern.search(line)
        if init_match:
            results['Init Time'] = (float(init_match.group(1)))

        # Match the inference timings
        inference_match = inference_pattern.search(line)
        if inference_match:
            results["Init Inference"] = int(inference_match.group(1))/1000
            results["First Inference"] = int(inference_match.group(2))/1000
            results["Warmup Inference"] = float(inference_match.group(3))/1000
            results["Avg Inference"] = float(inference_match.group(4))/1000

        # Match the memory footprint
        memory_match = memory_pattern.search(line)
        if memory_match:
            results['Memory Init'] = float(memory_match.group(1))
            results['Memory Overall'] = float(memory_match.group(2))

        # Match the Avg Inference STD
        inference_std = inference_std_pattern.search(line)
        if inference_std:
            results['Avg Inference STD'] = float(inference_std.group(1))/1000

def extract_results(results_dir, num_repetitions):
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Error: The input path '{results_dir}' does not exist.")
        return
    # Define model types (rows)
    rows = []
    for model in model_names:
        rows.append(model)
        rows.append(model + "_quant")

    # Define columns
    cols = []
    for metric in metrics:
        cols.append(metric)
    cols.append("Avg Inference STD")

    # Create an empty DataFrame to store results
    results_df = pd.DataFrame(index=rows, columns=cols)

    for model_name in rows:
        model_results = defaultdict(int)
        with open(f'{results_dir}/tflite_{model_name}_profiling.txt', 'r') as file:
            output_original = file.read()
        parse_benchmark_output(output_original, model_results, num_repetitions)

        for metric in metrics:
            results_df.loc[model_name, metric] = model_results[metric]
        results_df.loc[model_name, "Avg Inference STD"] = model_results["Avg Inference STD"]
    
    return results_df

def plot(results_df, save_dir):
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(f'{save_dir}/tflite_results.csv')
    for metric in metrics:
        means_orig = results_df.loc[model_names, metric].values
        means_quant = results_df.loc[[model + "_quant" for model in model_names], metric].values
        stds_orig = results_df.loc[model_names, "Avg Inference STD"].values if metric == "Avg Inference" else np.zeros_like(means_orig)
        stds_quant = results_df.loc[[model + "_quant" for model in model_names], "Avg Inference STD"].values if metric == "Avg Inference" else np.zeros_like(means_quant)

        n_groups = len(model_names)
        index = np.arange(n_groups)

        fig, ax = plt.subplots(figsize=(5, 4))
        bar_width = 0.35
        opacity = 0.8

        rects1 = plt.bar(index, means_orig, bar_width,
                        alpha=opacity,
                        label='Original')

        rects2 = plt.bar(index + bar_width, means_quant, bar_width,
                        alpha=opacity,
                        label='Quantized')

        if metric == "Avg Inference":
            plt.errorbar(index, means_orig, yerr=stds_orig, fmt='none', color='black', capsize=5)
            plt.errorbar(index + bar_width, means_quant, yerr=stds_quant, fmt='none', color='black', capsize=5)

        plt.xlabel('Model')
        if metric.startswith('Memory'):
            plt.ylabel(f'{metric} (MB)')
        else:
            plt.ylabel(f'{metric} Time (ms)')
        if metric.endswith('Inference'):
            plt.title(F'TFlite: {metric} Time')
        else:
            plt.title(F'TFlite: {metric}')
        plt.xticks(index + bar_width / 2, model_names, rotation=45)
        plt.legend()



        # Save the plot as an image
        # plt.subplots_adjust(left=0.15, right=0.95, bottom=0.25, top=0.95)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{metric}.png', format='png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', help='The directory where the TFlite models\' results are saved', required=True)
    parser.add_argument('--save_dir', help='The directory where the plots should be saved', required = True)
    parser.add_argument('--num_repetitions', type=int, help='The number of repetitions for each model', required = True)
    args = parser.parse_args()

    results_df = extract_results(args.results_dir, args.num_repetitions)
    plot(results_df, args.save_dir)
    print(results_df)