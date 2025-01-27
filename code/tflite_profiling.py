import argparse
import requests
import platform
import stat
import os
import subprocess



tflite_model_names = ["MobileNetV2", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]

def download_benchmark(dir, arch_type):
    url = f'https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_{arch_type}_benchmark_model'
    r = requests.get(url, allow_redirects=True)
    open(dir + '/benchmark', 'wb').write(r.content)
    os.chmod(dir + '/benchmark', stat.S_IEXEC)

def benchmark(results_dir, tflite_dir):
    results_dir = results_dir or '.'
    model_names = tflite_model_names + [f'{model_name}_quant' for model_name in tflite_model_names]
    for model_name in model_names:
        outputOriginal = subprocess.check_output(f'{results_dir}/benchmark --graph={tflite_dir}/{model_name}.tflite --num_threads=1 --num_runs=100 --enable_op_profiling=true > {results_dir}/tflite_{model_name}_profiling.txt', shell=True)
        outputOriginal = outputOriginal.decode('utf-8')
        print(outputOriginal)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite_dir', help='The directory where the TFlite models are saved', required=True)
    parser.add_argument('--results_dir', help='The directory to save the results')
    args = parser.parse_args()

    if not os.path.exists(args.tflite_dir):
        raise FileNotFoundError(f"Error: The input path '{args.tflite_dir}' does not exist.")
    
    arch_type = platform.machine().replace("_", "-")
    download_benchmark(args.results_dir, arch_type)
    benchmark(args.results_dir, args.tflite_dir)
