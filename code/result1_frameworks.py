import argparse
import matplotlib.pyplot as plt
import numpy as np
from tflite_operators import parse_results as get_tflite_operators
from onnx_operators import consolidate_results as get_onnx_operators
from tflite_plots import extract_results as extract_tflite_results
from onnx_plots import extract_results as extract_onnx_results
from aggregrate_operators import aggregate_convolution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model name', required=True)
    parser.add_argument('--tflite_result', help='The path where tflite results are stored', required=True)
    parser.add_argument('--onnx_result', help='The path where onnx results are stored', required= True)
    parser.add_argument('--num_repetitions', help='The number of times each model was profiled', required= True, type=int)
    parser.add_argument('--output', help='The name for the directory of the output plots', required=True)
    args = parser.parse_args()
    if args.model not in ["ResNet50"]:
        raise NotImplementedError("Currently, this code has not been extended for models other than ResNet50")
    
    #Load model run statistics
    tflite_orig_ops = get_tflite_operators(args.tflite_result + f'/tflite_{args.model}_profiling.txt')
    tflite_quant_ops = get_tflite_operators(args.tflite_result + f'/tflite_{args.model}_quant_profiling.txt')
    onnx_orig_ops = get_onnx_operators(args.onnx_result + f'/onnx_{args.model}_profiling', n = args.num_repetitions)
    onnx_quant_ops = get_onnx_operators(args.onnx_result + f'/onnx_{args.model}_quant_profiling', n = args.num_repetitions)
    tflite_df = extract_tflite_results(args.tflite_result, args.num_repetitions)
    onnx_df = extract_onnx_results(args.onnx_result, n = args.num_repetitions)

    frameworks = ['TFlite', 'ONNX']
    colors = ["#00cd63", "#339fff", "#f7b300"]

    #Part 1: Comapring Inference Times
    #Load inference times for the given model
    tflite_orig = tflite_df.loc[args.model, 'Avg Inference']
    tflite_quant = tflite_df.loc[args.model + '_quant', 'Avg Inference']
    onnx_orig = onnx_df.loc[args.model, 'model_run']
    onnx_quant = onnx_df.loc[args.model + '_quant', 'model_run']

    #Figure 1: Inference Time for Original and Quantized Models across TFlite and ONNX
    fig, plot1 = plt.subplots(figsize=(6, 4))
    bar_width = 0.25
    x = np.arange(len(frameworks))
    
    plot1.bar(x - bar_width / 2, [tflite_orig, onnx_orig], width=bar_width, label="Original", color=colors[0])
    plot1.bar(x + bar_width / 2, [tflite_quant, onnx_quant], width=bar_width, label="Quantized", color=colors[1])

    # Labels and title
    plot1.set_xticks(x)
    plot1.set_xticklabels(frameworks)
    plot1.set_ylabel('Time (ms)')
    plot1.set_title("Inference Time for ResNet50")
    plot1.legend()
    plt.savefig(f'{args.output}/Result1_{args.model}_InferenceTimes.png')

    
    #Part 2: Comparing time taken by convolution operator
    #Loading convolution times
    tflite_conv = aggregate_convolution("tflite", tflite_orig_ops, tflite_quant_ops, "ResNet50")
    onnx_conv = aggregate_convolution("onnx", onnx_orig_ops, onnx_quant_ops, "ResNet50") 
    categories = ['Convolution', 'Quantized Convolution', 'Convert']

    bar_width = 0.25
    fig, plot2 = plt.subplots(figsize=(6, 4))
    x = np.arange(len(frameworks))
    plot2.bar(x - bar_width / 2, [tflite_conv['Convolution'], onnx_conv['Convolution']], width=bar_width, label="Convolution", color=colors[0])

    # Plot stacked bars for Quantized Convolution + Convert
    plot2.bar(x + bar_width / 2, [tflite_conv['Quantized Convolution'], onnx_conv['Quantized Convolution']], width=bar_width, label='Quantized Convolution', color=colors[1])
    plot2.bar(x + bar_width / 2, [tflite_conv['Convert'], onnx_conv['Convert']], width=bar_width, label=categories[2], bottom=[tflite_conv['Quantized Convolution'], onnx_conv['Quantized Convolution']], color=colors[2])

    # Labels and title
    plot2.set_xticks(x)
    plot2.set_xticklabels(frameworks)
    plot2.set_ylabel('Time (ms)')
    plot2.set_title("Convolution Layer")
    plot2.legend()
    plt.savefig(f'{args.output}/Result1_{args.model}_ConvolutionComparison.png')



if __name__ == "__main__":
    main()
