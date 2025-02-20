import argparse
import matplotlib.pyplot as plt
import numpy as np
from tflite_operators import parse_results as get_tflite_operators
from onnx_operators import consolidate_results as get_onnx_operators
from tflite_plots import extract_results as extract_tflite_results
from onnx_plots import extract_results as extract_onnx_results
from aggregrate_operators import aggregate_convolution, aggregate_fullyconnected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite_result', help='The directory where tflite results are stored', required=True)
    parser.add_argument('--onnx_result', help='The directory where onnx results are stored', required= True)
    parser.add_argument('--num_repetitions', help='The number of times each model was profiled', required= True, type=int)
    parser.add_argument('--output', help='The name for the directory of the output plots', required=True)
    args = parser.parse_args()


    bar_width = 0.15 
    size = (2.5, 4)

    #TFLITE 
    for model in ["ResNet50", "MobileNetV2", "VGG16"]:
    
        tflite_orig_ops = get_tflite_operators(args.tflite_result + f'/tflite_{model}_profiling.txt')
        tflite_quant_ops = get_tflite_operators(args.tflite_result + f'/tflite_{model}_quant_profiling.txt')
        tflite_df = extract_tflite_results(args.tflite_result, args.num_repetitions)

        opacity = 0.8

        #Part 1: Comparing Inference Times for each Model
        tflite_orig = tflite_df.loc[model, 'Avg Inference']
        tflite_quant = tflite_df.loc[model + '_quant', 'Avg Inference']

        fig, plot1 = plt.subplots(figsize=size)

        x = np.arange(1)
        plot1.bar(x - bar_width / 2, tflite_orig, width=bar_width, label="Original", alpha = opacity)
        plot1.bar(x + bar_width / 2, tflite_quant, width=bar_width, label="Quantized", alpha = opacity)

        # Labels and title
        plot1.set_xticks(x)
        plot1.set_xticklabels([model])
        plot1.set_ylabel('Time (ms)')
        plot1.set_xlim(-0.7, 0.7)
        plot1.set_title(f"Inference Time \nfor {model}")
        plot1.legend()
        plt.tight_layout()
        plt.savefig(f'{args.output}/Result3_tflite_{model}_InferenceTimes.png')


        #Part 2: Comapring time taken by convolution operators across models
        tflite_conv = aggregate_convolution("tflite", tflite_orig_ops, tflite_quant_ops, model)
        fig, plot2 = plt.subplots(figsize=size)
        x = np.arange(1)
        if model == "MobileNetV2":
            plot2.bar(x - bar_width / 2, tflite_conv["Convolution"] + tflite_conv["Depthwise Convolution"], width=bar_width, label="Original Convolution", alpha = opacity)

            # Plot stacked bars for Quantized Convolution + Convert
            plot2.bar(x + bar_width / 2, tflite_conv["Quantized Convolution"] + tflite_conv["Quantized Depthwise Convolution"], width=bar_width, label="Quantized Convolution", alpha = opacity)
            plot2.bar(x + bar_width / 2, tflite_conv["Convert"] + tflite_conv["Pad"], width=bar_width, label="Convert", bottom=tflite_conv["Quantized Convolution"] + tflite_conv["Quantized Depthwise Convolution"], alpha = opacity)
        else:
            plot2.bar(x - bar_width / 2, tflite_conv["Convolution"], width=bar_width, label="Original \nConvolution", alpha = opacity)

            # Plot stacked bars for Quantized Convolution + Convert
            plot2.bar(x + bar_width / 2, tflite_conv["Quantized Convolution"], width=bar_width, label="Quantized \nConvolution", alpha = opacity)
            plot2.bar(x + bar_width / 2, tflite_conv["Convert"], width=bar_width, label="Convert", bottom=tflite_conv["Quantized Convolution"], alpha = opacity)

        # Labels and title
        plot2.set_xticks(x)
        #plot2.margins(x=1)
        plot2.set_xticklabels(['Convolution'], rotation=45)
        plot2.set_ylabel('Time (ms)')
        plot2.set_xlim(-0.2, 1.2)
        #plot2.set_title(f"Convolution Layer for {model}")
        if model == "VGG16":
            plot2.set_ylim(0, 548.1)
            plot2.legend()
        elif model == "ResNet50":
            plot2.set_ylim(0, 139.2)
        elif model == "MobileNetV2":
            plot2.set_ylim(0, 8.5)
        plot2.set_xlim(-0.7, 0.7)
        #plt.tight_layout()
        plt.subplots_adjust(left=0.25, right=0.95, bottom=0.3, top=0.95)
        plt.savefig(f'{args.output}/Result3_tflite_{model}_ConvolutionComparison.pdf', format='pdf')


        #Part 3: Comparing Fully Connected Layers across models
        tflite_fullyconnected = aggregate_fullyconnected("tflite", tflite_orig_ops, tflite_quant_ops, model)

        fig, plot3 = plt.subplots(figsize=size)
        x = np.arange(1)
        plot3.bar(x - bar_width / 2, tflite_fullyconnected["Fully Connected"], width=bar_width, label="Fully Connected", alpha = opacity)
        plot3.bar(x + bar_width / 2, tflite_fullyconnected["Quantized Fully Connected"], width=bar_width, label="Quantized \nFully Connected", alpha = opacity)

        # Labels and title
        plot3.set_xticks(x)
        plot3.set_xticklabels([model])
        plot3.set_ylabel('Time (ms)')
        plot3.set_xlim(-0.7, 0.7)
        plot3.set_title(f"Fully Connected Layer \nfor {model}")
        plot3.legend()
        plt.tight_layout()
        plt.savefig(f'{args.output}/Result3_tflite_{model}_FullyConnectedComparison.png')

    #ONNX 
    for model in ["ResNet50", "MobileNetV2", "VGG16"]:
    
        onnx_orig_ops = get_onnx_operators(args.onnx_result + f'/onnx_{model}_profiling', n = args.num_repetitions)
        onnx_quant_ops = get_onnx_operators(args.onnx_result + f'/onnx_{model}_quant_profiling', n = args.num_repetitions)
        onnx_df = extract_onnx_results(args.onnx_result, args.num_repetitions)

        #Part 1: Comparing Inference Times for each Model
        onnx_orig = onnx_df.loc[model, 'model_run']
        onnx_quant = onnx_df.loc[model + '_quant', 'model_run']

        fig, plot1 = plt.subplots(figsize=size)

        x = np.arange(1)
        plot1.bar(x - bar_width / 2, onnx_orig, width=bar_width, label="Original", alpha = opacity)
        plot1.bar(x + bar_width / 2, onnx_quant, width=bar_width, label="Quantized", alpha = opacity)

        # Labels and title
        plot1.set_xticks(x)
        plot1.set_xticklabels([model])
        plot1.set_ylabel('Time (ms)')
        plot1.set_title(f"Inference Time \nfor {model}")
        plot1.set_xlim(-0.7, 0.7)
        plot1.legend()
        plt.savefig(f'{args.output}/Result3_onnx_{model}_InferenceTimes.png')


        #Part 2: Comapring time taken by convolution operators across models
        onnx_conv = aggregate_convolution("onnx", onnx_orig_ops, onnx_quant_ops, model)
        fig, plot2 = plt.subplots(figsize=size)
        x = np.arange(1)
        plot2.bar(x - bar_width / 2, onnx_conv["Convolution"], width=bar_width, label="Convolution", alpha = opacity)

        # Plot stacked bars for Quantized Convolution + Convert
        plot2.bar(x + bar_width / 2, onnx_conv["Quantized Convolution"], width=bar_width, label="Quantized \nConvolution", alpha = opacity)
        plot2.bar(x + bar_width / 2, onnx_conv["Convert"], width=bar_width, label="Convert", bottom=onnx_conv["Quantized Convolution"], alpha = opacity)

        # Labels and title
        plot2.set_xticks(x)
        plot2.set_xticklabels([model])
        plot2.set_ylabel('Time (ms)')
        plot2.set_title(f"Convolution Layer \nfor {model}")
        plot2.set_xlim(-0.7, 0.7)
        plot2.legend()
        plt.tight_layout()
        plt.savefig(f'{args.output}/Result3_onnx_{model}_ConvolutionComparison.png')


        #Part 3: Comparing Fully Connected Layers across models
        onnx_fullyconnected = aggregate_fullyconnected("onnx", onnx_orig_ops, onnx_quant_ops, model)

        fig, plot3 = plt.subplots(figsize=size)
        x = np.arange(1)
        plot3.bar(x - bar_width / 2, onnx_fullyconnected["Fully Connected"], width=bar_width, label="Fully Connected", alpha = opacity)
        plot3.bar(x + bar_width / 2, onnx_fullyconnected["Quantized Fully Connected"], width=bar_width, label="Quantized \nFully Connected", alpha = opacity)

        # Labels and title
        plot3.set_xticks(x)
        plot3.set_xticklabels([model])
        plot3.set_xlim(-0.7, 0.7)
        plot3.set_ylabel('Time (ms)')
        plot3.set_title(f"Fully Connected Layer \nfor {model}")
        plot3.legend()
        plt.tight_layout()
        plt.savefig(f'{args.output}/Result3_onnx_{model}_FullyConnectedComparison.png')


if __name__ == "__main__":
    main()
