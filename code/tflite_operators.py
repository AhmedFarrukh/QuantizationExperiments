import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
import textwrap
from collections import defaultdict
import re
import json
import copy


ResNet50_matching = {
    "Convolution (NHWC, F32) IGEMM + Convolution (NHWC, F32) GEMM": ["Convolution (NHWC, QDU8, F32, QC8W) IGEMM", "Convolution (NHWC, QD8, F32, QC8W) IGEMM", "Convert (NC, F32, QDU8)", "Convert (NC, F32, QD8)"],
    "Unary Elementwise (NC)": ["Unary Elementwise (NC)"],
    "Binary Elementwise (ND)": ["Binary Elementwise (ND)"],
    "Fully Connected (NC, F32) GEMM": ["Fully Connected (NC, QDU8, F32, QC8W) GEMM", "Fully Connected (NC, QD8, F32, QC8W) GEMM"],
    "Constant Pad (ND, X32)": ["Constant Pad (ND, X32)"],
    "Max Pooling (NHWC, F32)": ["Max Pooling (NHWC, F32)"],
    "Mean (ND) Mean": ["Mean (ND) Mean"],
    "Softmax (NC, F32)": ["Softmax (NC, F32)"],
    #"Additional": []
    }

VGG16_matching = {
    "Convolution (NHWC, F32) IGEMM": ["Convolution (NHWC, QDU8, F32, QC8W) IGEMM", "Convolution (NHWC, QD8, F32, QC8W) IGEMM", "Convert (NC, F32, QDU8)", "Convert (NC, F32, QD8)"],
    "Fully Connected (NC, F32) GEMM": ["Fully Connected (NC, QDU8, F32, QC8W) GEMM", "Fully Connected (NC, QD8, F32, QC8W) GEMM"],
    "Max Pooling (NHWC, F32)": ["Max Pooling (NHWC, F32)"],
    "Softmax (NC, F32)": ["Softmax (NC, F32)"],
    "Copy (NC, X32)": ["Copy (NC, X32)"]
    #"Additional": []
    }

MobileNetV2_matching = {
    "Convolution (NHWC, F32) GEMM":["Convolution (NHWC, QDU8, F32, QC8W) IGEMM", "Convolution (NHWC, QD8, F32, QC8W) IGEMM", "Convolution (NHWC, F32) GEMM", "Convert (NC, F32, QDU8)", "Convert (NC, F32, QD8)"],
    "Convolution (NHWC, F32) DWConv":["DEPTHWISE_CONV_2D", "Convolution (NHWC, F32) DWConv", "Constant Pad (ND, X32)"],
    "Convolution (NHWC, F32) IGEMM":["Convolution (NHWC, F32) IGEMM"],
    "Fully Connected (NC, F32) GEMM":["Fully Connected (NC, QDU8, F32, QC8W) GEMM", "Fully Connected (NC, QD8, F32, QC8W) GEMM"],
    "Binary Elementwise (ND)":["Binary Elementwise (ND)"],
    "Mean (ND) Mean":["Mean (ND) Mean"],
    "Softmax (NC, F32)":["Softmax (NC, F32)"],
    }
    
def plot(orig_ops, quant_ops, output_name, model):
    if model == "ResNet50":
        matching = ResNet50_matching
    elif model == "VGG16":
        matching = VGG16_matching
    elif model == "MobileNetV2":
        matching = MobileNetV2_matching
    else:
        raise NotImplementedError
    # Prepare data for the first plot (Original Operations)
    orig_operations = [operation for operation in orig_ops]
    orig_durations = [orig_ops[operation]["duration"] for operation in orig_operations]

    # Plot the first horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(np.arange(len(orig_operations)), orig_durations, color="#1f77b4", label='Original')

    # Customize the first graph
    plt.title(f"TFlite-{model}-Original", loc='center')
    plt.xlabel('Average Duration - ms')
    plt.ylabel('Operation Types')
    plt.yticks(np.arange(len(orig_operations)), orig_operations)
    plt.legend()

    # Save the first plot
    plt.tight_layout()
    plt.savefig(output_name + "_original.png")
    plt.close()

    # Prepare data for the second plot (Stacked Quantized Operations)
    matching_operations = list(matching.keys())  # Use only matching dictionary keys
    n_operations = len(matching_operations)
    stack_durations = np.zeros(n_operations)  # Initialize baseline for stacking

    plt.figure(figsize=(12, 8))

    # Define manually distinct colors
    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
        "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
        "#dbdb8d", "#9edae5"
    ]
    color_index = 0

    quant_ops_plotted = set()

    # Use matching dictionary to stack equivalent operators
    for op_idx, op_type in enumerate(matching_operations):
        for eq_op in matching[op_type]:
            if eq_op in quant_ops:
                # Directly plot the duration for the matching quantized operation
                plt.barh(
                    op_idx,  # Specify the index for this operation
                    quant_ops[eq_op]["duration"],  # Use the duration value directly
                    left=stack_durations[op_idx],  # Use the existing baseline for stacking
                    color=color_palette[color_index % len(color_palette)],  # Assign unique color
                    label=eq_op
                )
                stack_durations[op_idx] += quant_ops[eq_op]["duration"]  # Update baseline for this row
                color_index += 1  # Increment color index
                quant_ops_plotted.add(eq_op)
    
    if set(quant_ops) - quant_ops_plotted:
        print("ERROR: THE FOLLOWING OPERATIONS WERE NOT MAPPED!")
        print(set(quant_ops) - quant_ops_plotted)


    # Overlay red markers for original operator durations
    orig_ops_matching = copy.deepcopy(orig_ops)
    if model == "ResNet50":
        del orig_ops_matching["Convolution (NHWC, F32) IGEMM"]
        del orig_ops_matching["Convolution (NHWC, F32) GEMM"]
        orig_ops_matching["Convolution (NHWC, F32) IGEMM + Convolution (NHWC, F32) GEMM"] = {"duration":0, "count":0}
        orig_ops_matching["Convolution (NHWC, F32) IGEMM + Convolution (NHWC, F32) GEMM"]["duration"] = orig_ops["Convolution (NHWC, F32) IGEMM"]["duration"] + orig_ops["Convolution (NHWC, F32) GEMM"]["duration"]
        orig_ops_matching["Convolution (NHWC, F32) IGEMM + Convolution (NHWC, F32) GEMM"]["count"] = orig_ops["Convolution (NHWC, F32) IGEMM"]["count"] + orig_ops["Convolution (NHWC, F32) GEMM"]["count"]

    for op_idx, op_type in enumerate(matching_operations):
        if op_type in orig_ops_matching:
            # Place a red marker above the total duration for this operation
            plt.plot(
                orig_ops_matching[op_type]["duration"],  # x-coordinate (duration)
                op_idx,  # y-coordinate (operation index)
                marker="o", color="red", markersize=8, label=None  # Red marker
            )

    # Customize the second graph
    plt.title(f"TFlite-{model}-Quantized", loc='center')
    plt.xlabel('Average Duration - ms')
    plt.ylabel('Operation Types')
    plt.yticks(np.arange(n_operations), matching_operations)  # Only use matching keys for labels
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend below the plot

    # Save the second plot
    plt.tight_layout()
    plt.savefig(output_name + "_quantized.png", bbox_inches='tight')
    plt.close()



def parse_results(result_path):
    # Define the markers for the section of interest
    section_start = r"INFO: Operator-wise Profiling Info for Regular Benchmark Runs:"
    table_start = r"============================== Summary by node type =============================="
    table_end = r"Timings \(microseconds\):"
    
    # Read the file content
    with open(result_path, 'r') as file:
        content = file.read()
    
    # Extract the block containing the table
    match = re.search(f"{section_start}(.*?){table_start}(.*?){table_end}", content, re.DOTALL)
    if not match:
        raise ValueError("Table not found in the file")
    
    table_block = match.group(2).strip()
    
    # Parse the table into a dictionary
    lines = table_block.split("\n")
    headers = re.split(r"\s{2,}", lines[0].strip())  # Use whitespace to split headers
    data_lines = lines[1:]  # The rest are data rows
    
    # Create a dictionary for operations
    ops = {}
    for line in data_lines:
        # Split line into fields using whitespace
        fields = re.split(r"\s{2,}", line.strip())
        
        # Extract fields based on known order
        op_name = fields[0]
        count = int(fields[1])
        avg_ms = float(fields[2])
        
        # Add to the dictionary
        ops[op_name] = {"count": count, "duration": avg_ms}
    
    for op in ops:
        print(f"Operator: {op}, Duration: {ops[op]['duration']}, Count: {ops[op]['count']}")

    return ops


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model name', required=True)
    parser.add_argument('--orig_result_path', help='The path of the txt file with the result from the original model', required=True)
    parser.add_argument('--quant_result_path', help='The path of the txt file with the result from the quantized model', required=True)
    parser.add_argument('--output_name', help='The name for the output plots', required=True)
    args = parser.parse_args()
    if args.model not in ["ResNet50", "VGG16", "MobileNetV2"]:
        raise NotImplementedError("Currently, this code has not been extended for models other than ResNet50, VGG16 and MobileNetV2")
    print(f"Original Model - {args.model}:")
    orig_ops = parse_results(args.orig_result_path)
    print(f"Ouantized Model - {args.model}:")
    quant_ops = parse_results(args.quant_result_path)
    plot(orig_ops, quant_ops, args.output_name, args.model) 
