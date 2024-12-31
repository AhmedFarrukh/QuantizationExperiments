import onnxruntime as ort
import numpy as np
import argparse
import os
import json
import pathlib
import matplotlib.pyplot as plt
import pandas as pd

model_names = ["MobileNet", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]
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

    # Create an empty DataFrame to store results
    results_df = pd.DataFrame(0.0, index=rows, columns=cols)

    for model in rows:
        for i in range(num_repetitions):
            result_path = os.path.join(results_dir, f"onnx_{model}_profiling_{i}.json")
            with open(result_path, 'r') as file:
                data = json.load(file)

            for entry in data:
                if entry.get('name') in metrics:
                    results_df.loc[model, entry.get('name')] += entry['dur']

    results_df = results_df / (num_repetitions*1000) #average and convert from us to ms 

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
        plt.title(F'ONNX: {titles[metric]}')
        plt.xticks(index + bar_width / 2, model_names, rotation=45)
        plt.legend()

        plt.tight_layout()

        # Save the plot as an image
        plt.savefig(f'{save_dir}/{metric}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', help='The directory where the ONNX models\' results are saved', required=True)
    parser.add_argument('--save_dir', help='The directory where the plots should be saved', required = True)
    parser.add_argument('--num_repetitions', type=int, help='The number of profiling files for each model', required = True)
    args = parser.parse_args()

    results_df = extract_results(args.results_dir, args.num_repetitions)
    plot(results_df, args.save_dir)
    print(results_df)