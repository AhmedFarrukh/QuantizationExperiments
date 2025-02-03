import numpy as np
import argparse
from tflite_plots import extract_results as extract_tflite_results
from onnx_plots import extract_results as extract_onnx_results
from pytorch_plots import extract_results as extract_pytorch_results
import pathlib
import matplotlib.pyplot as plt

model_names = ["MobileNetV2", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite_dir', help='The directory where the TFlite models\' results are saved', required=True)
    parser.add_argument('--onnx_dir', help='The directory where the ONNX models\' results are saved', required=True)
    parser.add_argument('--pytorch_dir', help='The directory where the PyTorch models\' results are saved', required=True)
    parser.add_argument('--onnx_num_repetitions', type=int, help='The number of profiling files for each ONNX model', required = True)
    parser.add_argument('--pytorch_num_repetitions', type=int, help='The number of profiling files for each PyTorch model', required = True)
    parser.add_argument('--save_dir', help='The directory where the plots should be saved', required = True)

    args = parser.parse_args()

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    tflite_df = extract_tflite_results(args.tflite_dir)
    onnx_df = extract_onnx_results(args.onnx_dir, args.onnx_num_repetitions)
    pytorch_df = extract_pytorch_results(args.pytorch_dir, args.pytorch_num_repetitions)

    tflite_orig = tflite_df.loc[model_names, 'Avg Inference'].values
    tflite_quant = tflite_df.loc[[model + "_quant" for model in model_names], 'Avg Inference'].values

    onnx_orig = onnx_df.loc[model_names, 'model_run'].values
    onnx_quant = onnx_df.loc[[model + "_quant" for model in model_names], 'model_run'].values
    
    pytorch_orig = pytorch_df.loc[model_names, "CPU time total"].values
    pytorch_quant = pytorch_df.loc[[model + "_quant" for model in model_names], "CPU time total"].values

    #Graph of Original Inference Times
    n_groups = len(model_names)
    index = np.arange(n_groups)

    fig, ax = plt.subplots()
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, tflite_orig, bar_width,
                    alpha=opacity,
                    label='TFlite')

    rects2 = plt.bar(index + bar_width, onnx_orig, bar_width,
                    alpha=opacity,
                    label='ONNX')

    rects3 = plt.bar(index + bar_width, pytorch_orig, bar_width,
                    alpha=opacity,
                    label='PyTorch')

    plt.xlabel('Model')
    plt.ylabel(f'Inference Time (ms)')
    plt.title(f'Comparing Original Inference Times')
    plt.xticks(index + bar_width / 2, model_names, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{save_dir}/OrigInferenceTimes.png')


    #Graph of Quantized Inference Times
    fig, ax = plt.subplots()
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, tflite_quant, bar_width,
                    alpha=opacity,
                    label='TFlite')

    rects2 = plt.bar(index + bar_width, onnx_quant, bar_width,
                    alpha=opacity,
                    label='ONNX')

    rects3 = plt.bar(index + bar_width, pytorch_quant, bar_width,
                    alpha=opacity,
                    label='PyTorch')

    plt.xlabel('Model')
    plt.ylabel(f'Inference Time (ms)')
    plt.title(f'Comparing Inference Times Post-Quantization')
    plt.xticks(index + bar_width / 2, model_names, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{save_dir}/QuantInferenceTimes.png')
