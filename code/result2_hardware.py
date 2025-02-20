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
    parser.add_argument('--num_repetitions', help='The number of times each model was profiled', required= True, type=int)
    parser.add_argument('--output', help='The name for the directory of the output plots', required=True)
    args = parser.parse_args()
    if args.model not in ["ResNet50"]:
        raise NotImplementedError("Currently, this code has not been extended for models other than ResNet50")
    
    hardware_lst = ["compute_haswell_ib", "compute_skylake", "compute_icelake_r650", "compute_cascadelake"]
    hardware_names = ["Haswell", "Skylake", "Icelake", "Cascade Lake"]
    tflite_orig_inference = []
    tflite_quant_inference = []
    tflite_orig_inference_std = []
    tflite_quant_inference_std = []
    tflite_orig_conv = []
    tflite_quant_conv = []
    tflite_quant_convert = []

    for hardware in hardware_lst:
        #Load Inference Time Statistics
        tflite_orig_ops = get_tflite_operators(args.tflite_result + f'/{hardware}/tflite_profiling_results/tflite_{args.model}_profiling.txt')
        tflite_quant_ops = get_tflite_operators(args.tflite_result + f'/{hardware}/tflite_profiling_results/tflite_{args.model}_quant_profiling.txt')
        tflite_df = extract_tflite_results(args.tflite_result + f'/{hardware}/tflite_profiling_results', args.num_repetitions)

        #Load required statistics for plots
        tflite_conv = aggregate_convolution("tflite", tflite_orig_ops, tflite_quant_ops, "ResNet50")

        tflite_orig_inference.append(tflite_df.loc[args.model, 'Avg Inference'])
        tflite_quant_inference.append(tflite_df.loc[args.model + '_quant', 'Avg Inference'])
        tflite_orig_inference_std.append(tflite_df.loc[args.model, 'Avg Inference STD'])
        tflite_quant_inference_std.append(tflite_df.loc[args.model + '_quant', 'Avg Inference STD'])
        tflite_orig_conv.append(tflite_conv["Convolution"])
        tflite_quant_conv.append(tflite_conv["Quantized Convolution"])
        tflite_quant_convert.append(tflite_conv["Convert"])

    bar_width = 0.35
    opacity = 0.8
    x = np.arange(len(hardware_lst))

    #Part 1: Overall Inference times before and after quantization across hardware
    fig, ax = plt.subplots(figsize=(3, 4))

    rects1 = ax.bar(x - bar_width / 2, tflite_orig_inference, width=bar_width, label="Original", alpha = opacity)
    rects2 = ax.bar(x + bar_width / 2, tflite_quant_inference, width=bar_width, label="Quantized", alpha = opacity)

    plt.errorbar(x - bar_width / 2, tflite_orig_inference, yerr=tflite_orig_inference_std, fmt='none', color='black', capsize=5)
    plt.errorbar(x + bar_width / 2, tflite_quant_inference, yerr=tflite_quant_inference_std, fmt='none', color='black', capsize=5)

    # Labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(hardware_names, rotation=45)
    ax.set_ylabel('Time (ms)')
    ax.set_xlabel('Hardware')
    #plot1.set_title(f"Inference Time for {args.model}")
    ax.legend()
    #plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.3, top=0.95)
    plt.savefig(f'{args.output}/Result2_{args.model}_InferenceTimes.pdf', format='pdf')

    y_range = ax.get_ylim()

    #Part2: Comparing time taken by convolution operator across hardware
    bar_width = 0.35
    fig, ax2 = plt.subplots(figsize=(3, 4))
    
    ax2.bar(x - bar_width / 2, tflite_orig_conv, width=bar_width, label="Convolution", alpha = opacity)

    # Plot stacked bars for Quantized Convolution + Convert
    ax2.bar(x + bar_width / 2, tflite_quant_conv, width=bar_width, label="Quantized \nConvolution", alpha = opacity)
    ax2.bar(x + bar_width / 2, tflite_quant_convert, width=bar_width, label="Convert", bottom=tflite_quant_conv, alpha = opacity)

    # Labels and title
    ax2.set_xticks(x)
    ax2.set_xticklabels(hardware_names, rotation=45)
    ax2.set_ylim(y_range)
    #ax2.set_ylabel('Time (ms)')
    ax2.set_xlabel('Hardware')
    #plot2.set_title(f"Convolution Layer for {args.model}")
    ax2.legend()
    ax2.set_yticklabels([])  # Also removes the tick labels
    #plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.3, top=0.95)
    plt.savefig(f'{args.output}/Result2_{args.model}_ConvolutionComparison.pdf', format='pdf')


if __name__ == "__main__":
    main()