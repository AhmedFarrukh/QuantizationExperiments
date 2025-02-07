import argparse
import matplotlib.pyplot as plt
import numpy as np
from tflite_operators import parse_results as get_tflite_operators
from tflite_plots import extract_results as extract_tflite_results
from aggregrate_operators import aggregate_convolution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model name', required=True)
    parser.add_argument('--tflite_result', help='The parent directory with subdirectories containing hardware-specific results', required=True)
    parser.add_argument('--output', help='The name for the directory of the output plots', required=True)
    args = parser.parse_args()
    if args.model not in ["ResNet50"]:
        raise NotImplementedError("Currently, this code has not been extended for models other than ResNet50")
    
    hardware_lst = ["compute_haswell_ib", "compute_skylake", "compute_icelake", "compute_cascadelake_r"]
    hardware_names = ["Haswell", "Skylake", "Icelake", "Cascade Lake"]
    tflite_orig_inference = []
    tflite_quant_inference = []
    tflite_orig_conv = []
    tflite_quant_conv = []
    tflite_quant_convert = []

    for hardware in hardware_lst:
        #Load Inference Time Statistics
        tflite_orig_ops = get_tflite_operators(args.tflite_result + f'/{hardware}/tflite_profiling_results/tflite_{args.model}_profiling.txt')
        tflite_quant_ops = get_tflite_operators(args.tflite_result + f'/{hardware}/tflite_profiling_results/tflite_{args.model}_quant_profiling.txt')
        tflite_df = extract_tflite_results(args.tflite_result + f'/{hardware}/tflite_profiling_results')

        #Load required statistics for plots
        tflite_conv = aggregate_convolution("tflite", tflite_orig_ops, tflite_quant_ops, "ResNet50")

        tflite_orig_inference.append(tflite_df.loc[args.model, 'Avg Inference'])
        tflite_quant_inference.append(tflite_df.loc[args.model + '_quant', 'Avg Inference'])
        tflite_orig_conv.append(tflite_conv["Convolution"])
        tflite_quant_conv.append(tflite_conv["Quantized Convolution"])
        tflite_quant_convert.append(tflite_conv["Convert"])

    bar_width = 0.25
    colors = ["#00cd63", "#339fff", "#f7b300"]
    x = np.arange(len(hardware_lst))

    #Part 1: Overall Inference times before and after quantization across hardware
    fig, plot1 = plt.subplots(figsize=(6, 4))
    bar_width = 0.25

    plot1.bar(x - bar_width / 2, tflite_orig_inference, width=bar_width, label="Original", color=colors[0])
    plot1.bar(x + bar_width / 2, tflite_quant_inference, width=bar_width, label="Quantized", color=colors[1])

    # Labels and title
    plot1.set_xticks(x)
    plot1.set_xticklabels(hardware_names)
    plot1.set_ylabel('Time (ms)')
    plot1.set_title(f"Inference Time for {args.model}")
    plot1.legend()
    plt.savefig(f'{args.output}/Result2_{args.model}_InferenceTimes.png')

    #Part2: Comparing time taken by convolution operator across hardware
    fig, plot2 = plt.subplots(figsize=(6, 4))
    
    plot2.bar(x - bar_width / 2, tflite_orig_conv, width=bar_width, label="Convolution", color=colors[0])

    # Plot stacked bars for Quantized Convolution + Convert
    plot2.bar(x + bar_width / 2, tflite_quant_conv, width=bar_width, label="Quantized Convolution", color=colors[1])
    plot2.bar(x + bar_width / 2, tflite_quant_convert, width=bar_width, label="Convert", bottom=tflite_quant_conv, color=colors[2])

    # Labels and title
    plot2.set_xticks(x)
    plot2.set_xticklabels(hardware_names)
    plot2.set_ylabel('Time (ms)')
    plot2.set_title("Convolution Layer")
    plot2.legend()
    plt.savefig(f'{args.output}/Result2_{args.model}_ConvolutionComparison.png')


if __name__ == "__main__":
    main()