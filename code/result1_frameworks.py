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
    onnx_df = extract_onnx_results(args.onnx_result, args.num_repetitions)

    frameworks = ['TFlite', 'ONNX']

    #Part 1: Comapring Inference Times
    #Load inference times for the given model
    tflite_orig = tflite_df.loc[args.model, 'Avg Inference']
    tflite_orig_error = tflite_df.loc[args.model, 'Avg Inference STD']
    tflite_quant = tflite_df.loc[args.model + '_quant', 'Avg Inference']
    tflite_quant_error = tflite_df.loc[args.model + '_quant', 'Avg Inference STD']
    onnx_orig = onnx_df.loc[args.model, 'model_run']
    onnx_orig_error = onnx_df.loc[args.model, 'model_run_sd']
    onnx_quant = onnx_df.loc[args.model + '_quant', 'model_run']
    onnx_quant_error = onnx_df.loc[args.model + '_quant', 'model_run_sd']
    
    #Figure 1: Inference Time for Original and Quantized Models across TFlite and ONNX

    fig, ax = plt.subplots(figsize=(2.5, 4))
    bar_width = 0.35
    opacity = 0.8

    x = np.arange(len(frameworks))

    # Plot the bars
    rects1 = ax.bar(
        x - bar_width/2,
        [tflite_orig, onnx_orig],
        bar_width,
        alpha=opacity,
        label='Original'
    )
    rects2 = ax.bar(
        x + bar_width/2,
        [tflite_quant, onnx_quant],
        bar_width,
        alpha=opacity,
        label='Quantized'
    )

    # Plot the error bars separately
    ax.errorbar(
        x - bar_width/2,
        [tflite_orig, onnx_orig],
        yerr=[tflite_orig_error, onnx_orig_error],
        fmt='none',           
        ecolor='black',       
        capsize=5,            
        elinewidth=1
    )
    ax.errorbar(
        x + bar_width/2,
        [tflite_quant, onnx_quant],
        yerr=[tflite_quant_error, onnx_quant_error],
        fmt='none',
        ecolor='black',
        capsize=5,
        elinewidth=1
    )

    ax.set_xlim(-1.2, 2.2)
    ax.set_ylim(0, 120)
    ax.set_xticks(x)
    ax.set_xticklabels(frameworks, rotation=45)
    ax.set_ylabel('Time (ms)')
    #ax.set_title(f'Inference Time for {args.model}')
    ax.legend()
    ax.set_xlabel('Framework')

    y_range = ax.get_ylim()
    x_range = ax.get_xlim()

    #plt.tight_layout()
    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.25, top=0.95)
    plt.savefig(f'{args.output}/Result1_{args.model}_InferenceTimes.pdf', format='pdf')
    
    #Part 2: Comparing time taken by convolution operator
    #Loading convolution times
    tflite_conv = aggregate_convolution("tflite", tflite_orig_ops, tflite_quant_ops, "ResNet50")
    onnx_conv = aggregate_convolution("onnx", onnx_orig_ops, onnx_quant_ops, "ResNet50") 
    categories = ['Convolution', 'Quantized Convolution', 'Convert']

    fig, ax = plt.subplots(figsize=(2.5, 4))
    x = np.arange(len(frameworks))
    
    rects1 = ax.bar(x - bar_width / 2, 
                    [tflite_conv['Convolution'], onnx_conv['Convolution']], 
                    width=bar_width, 
                    label="Convolution", 
                    alpha = opacity)

    # Plot stacked bars for Quantized Convolution + Convert
    rects2 = ax.bar(x + bar_width / 2, 
                    [tflite_conv['Quantized Convolution'], onnx_conv['Quantized Convolution']], 
                    width=bar_width, 
                    label='Quantized \nConvolution', 
                    alpha = opacity)
    rects3 = ax.bar(x + bar_width / 2, 
                    [tflite_conv['Convert'], onnx_conv['Convert']], 
                    width=bar_width, 
                    label='Convert', 
                    bottom=[tflite_conv['Quantized Convolution'], onnx_conv['Quantized Convolution']], 
                    alpha = opacity)

    # Labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(frameworks, rotation=45)
    #ax.set_ylabel('Time (ms)')
    ax.set_xlabel('Framework')
    #ax.set_title(f"Convolution Layer for {args.model}")
    ax.legend()
    ax.set_ylim(y_range)
    ax.set_xlim(x_range)
    ax.set_yticklabels([])  # Also removes the tick labels
    #plt.tight_layout()
    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.25, top=0.95)
    plt.savefig(f'{args.output}/Result1_{args.model}_ConvolutionComparison.pdf', format='pdf')



if __name__ == "__main__":
    main()
