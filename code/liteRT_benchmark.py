import argparse
import pathlib
import gdown
import requests
import platform
import sys
import stat
import os
import pandas as pd
import re
import subprocess
from collections import defaultdict
from statistics import mean
from statistics import stdev
import matplotlib.pyplot as plt
import numpy as np



def download_models(dir, url):
    tflite_models_dir = pathlib.Path(dir + '/models')
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    gdown.download_folder(url, output = dir + '/models')

def download_benchmark(dir, arch_type):
    url = f'https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_{arch_type}_benchmark_model'
    r = requests.get(url, allow_redirects=True)
    open(dir + '/benchmark', 'wb').write(r.content)
    os.chmod(dir + '/benchmark', stat.S_IEXEC)

def benchmark(dir, modelNames, metrics):
    # Define model types (rows)
    rows = []
    for model in modelNames:
        rows.append(model)
        rows.append(model + "_quant")

    # Define columns
    cols = []
    for metric in metrics:
        cols.append(metric)
        cols.append(metric + "_sd")

    # Create an empty DataFrame to store results
    finalResult = pd.DataFrame(index=rows, columns=cols)
    n = 10 #the number of times the benchmark is called for each model

    for modelName in rows:
        print(modelName)
        modelResults = defaultdict(list)
        for i in range(n):
            outputOriginal = subprocess.check_output(f'{dir}/benchmark --graph={dir}/models/{modelName}.tflite --num_threads=1', shell=True)
            outputOriginal = outputOriginal.decode('utf-8')
            print(outputOriginal)
            parse_benchmark_output(outputOriginal, modelResults)

        for metric in metrics:
            finalResult.loc[modelName, metric] = mean(modelResults[metric])
            finalResult.loc[modelName, metric + "_sd"] = stdev(modelResults[metric])
    
    return finalResult
        

def parse_benchmark_output(output, results):
    """
    Parse benchmark output to extract model initialization times, inference timings, and memory footprint.
    """

    # Regular expressions to match the required information
    init_time_patterns = [
        re.compile(r'INFO: Initialized session in (\d+.\d+)ms.'),
        re.compile(r'INFO: Initialized session in (\d+)ms.')
    ]
    inference_patterns = [
        re.compile(r'INFO: Inference timings in us: Init: (\d+), First inference: (\d+), Warmup \(avg\): ([\d.e+]+), Inference \(avg\): ([\d.e+]+)'),
        re.compile(r'INFO: Inference timings in us: Init: (\d+), First inference: (\d+), Warmup \(avg\): ([\d.e+]+), Inference \(avg\): (\d+)'),
        re.compile(r'INFO: Inference timings in us: Init: (\d+), First inference: (\d+), Warmup \(avg\): (\d+.\d+), Inference \(avg\): (\d+.\d+)'),
        re.compile(r'INFO: Inference timings in us: Init: (\d+), First inference: (\d+), Warmup \(avg\): (\d+), Inference \(avg\): (\d+.\d+)'),
        re.compile(r'INFO: Inference timings in us: Init: (\d+), First inference: (\d+), Warmup \(avg\): (\d+), Inference \(avg\): (\d+)'),
    ]
    memory_patterns = [
        re.compile(r'INFO: Memory footprint delta from the start of the tool \(MB\): init=(\d+.\d+) overall=(\d+.\d+)'),
        re.compile(r'INFO: Memory footprint delta from the start of the tool \(MB\): init=(\d+.\d+) overall=(\d+)'),
        re.compile(r'INFO: Memory footprint delta from the start of the tool \(MB\): init=(\d+) overall=(\d+.\d+)'),
        re.compile(r'INFO: Memory footprint delta from the start of the tool \(MB\): init=(\d+) overall=(\d+)'),
    ]
    for line in output.split('\n'):
        # Match the initialization time
        for pattern in init_time_patterns:
            init_match = pattern.search(line)
            if init_match:
                results['Init Time (ms)'].append(float(init_match.group(1)))
                break

        # Match the inference timings
        for pattern in inference_patterns:
            inference_match = pattern.search(line)
            if inference_match:
                results["Init Inference (ms)"].append(int(inference_match.group(1))/1000)
                results["First Inference (ms)"].append(int(inference_match.group(2))/1000)
                results["Warmup Inference (ms)"].append(float(inference_match.group(3))/1000)
                results["Avg Inference (ms)"].append(float(inference_match.group(4))/1000)
                break

        # Match the memory footprint
        for pattern in memory_patterns:
            memory_match = pattern.search(line)
            if memory_match:
              results['Memory Init (MB)'].append(float(memory_match.group(1)))
              results['Memory Overall (MB)'].append(float(memory_match.group(2)))
              break

def print_results(benchmark_results, metrics, modelNames, dir, name):
    results_dir = pathlib.Path(dir + '/results')
    results_dir.mkdir(exist_ok=True, parents=True)
    for metric in metrics:
        means_orig = benchmark_results.loc[modelNames, metric].values
        errors_orig = benchmark_results.loc[modelNames, metric + "_sd"].values
        means_quant = benchmark_results.loc[[model + "_quant" for model in modelNames], metric].values
        errors_quant = benchmark_results.loc[[model + "_quant" for model in modelNames], metric + "_sd"].values


        n_groups = len(modelNames)
        index = np.arange(n_groups)

        fig, ax = plt.subplots()
        bar_width = 0.35
        opacity = 0.8

        rects1 = plt.bar(index, means_orig, bar_width,
                        alpha=opacity,
                        yerr=errors_orig,
                        label='Original')

        rects2 = plt.bar(index + bar_width, means_quant, bar_width,
                        alpha=opacity,
                        yerr=errors_quant,
                        label='Quantized')

        plt.xlabel('Model')
        plt.ylabel(metric)
        if name:
            plt.title(f'Bar Chart for {metric} on {name}')
        else:
            plt.title(f'Bar Chart for {metric}')
        plt.xticks(index + bar_width / 2, modelNames, rotation=45)
        plt.legend()

        plt.tight_layout()

        # Save the plot as an image
        plt.savefig(f'{results_dir}/{metric}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='The directory to save the models and results', required=True)
    parser.add_argument('--gdrive_link', help='The google drive URL where the models (quantized and original) can be accessed', required=True)
    parser.add_argument('--name', help='The device name/type for labelling the graphs', required=False)
    
    args = parser.parse_args()
    
    arch_type = platform.machine().replace("_", "-")
    modelNames = ["MobileNetV2", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]
    metrics = ["Init Time (ms)", "Init Inference (ms)", "First Inference (ms)", "Warmup Inference (ms)", "Avg Inference (ms)", "Memory Init (MB)", "Memory Overall (MB)"]

    download_models(args.dir, args.gdrive_link)

    download_benchmark(args.dir, arch_type)
    
    benchmark_result = benchmark(args.dir, modelNames, metrics)

    print_results(benchmark_result, metrics, modelNames, args.dir, args.name)
