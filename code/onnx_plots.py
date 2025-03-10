import numpy as np
import argparse
import os
import json
import pathlib
import matplotlib.pyplot as plt
import pandas as pd

model_names = ["MobileNetV2", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]
metrics = ["model_loading_uri", "session_initialization", "model_run"]

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
    
    cols.append("model_run_sd")

    # Create an empty DataFrame to store results
    results_df = pd.DataFrame(0.0, index=rows, columns=cols)

    for model in rows:
        model_run_times = []
        for i in range(num_repetitions):
            result_path = os.path.join(results_dir, f"onnx_{model}_profiling_{i}.json")
            with open(result_path, 'r') as file:
                data = json.load(file)

            for entry in data:
                if entry.get('name') in metrics:
                    results_df.loc[model, entry.get('name')] += entry['dur']
                if entry.get('name') == "model_run":
                    model_run_times.append(entry['dur'])
        results_df.loc[model, "model_run_sd"] = np.std(model_run_times)

    # Scale all columns except "model_run_sd"
    results_df.loc[:, results_df.columns != "model_run_sd"] /= (num_repetitions * 1000)

    # Scale only the "model_run_sd" column by 1000
    results_df.loc[:, "model_run_sd"] /= 1000


    return results_df

def plot(results_df, save_dir):
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(f'{save_dir}/onnx_results.csv')

    titles = {"model_loading_uri": "Model Loading Time", 
              "session_initialization": "Session Initialization Time", 
              "model_run": "Avg Inference Time"}

    for metric in metrics:
        means_orig = results_df.loc[model_names, metric].values
        means_quant = results_df.loc[[model + "_quant" for model in model_names], metric].values
        stds_orig = results_df.loc[model_names, "model_run_sd"].values if metric == "model_run" else np.zeros_like(means_orig)
        stds_quant = results_df.loc[[model + "_quant" for model in model_names], "model_run_sd"].values if metric == "model_run" else np.zeros_like(means_quant)

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
        
        if metric == "model_run":
            plt.errorbar(index, means_orig, yerr=stds_orig, fmt='none', color='black', capsize=5)
            plt.errorbar(index + bar_width, means_quant, yerr=stds_quant, fmt='none', color='black', capsize=5)

        plt.xlabel('Model')
        plt.ylabel(f'{titles[metric]} (ms)')
        #plt.title(F'ONNX: {titles[metric]}')
        plt.xticks(index + bar_width / 2, model_names, rotation=45)
        plt.legend()

        plt.tight_layout()

        # Save the plot as an image
        plt.savefig(f'{save_dir}/{metric}.png')

        # plt.subplots_adjust(left=0.15, right=0.95, bottom=0.25, top=0.95)
        # plt.savefig(f'{save_dir}/{metric}.pdf', format='pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', help='The directory where the ONNX models\' results are saved', required=True)
    parser.add_argument('--save_dir', help='The directory where the plots should be saved', required = True)
    parser.add_argument('--num_repetitions', type=int, help='The number of profiling files for each model', required = True)
    args = parser.parse_args()

    results_df = extract_results(args.results_dir, args.num_repetitions)
    plot(results_df, args.save_dir)
    print(results_df)