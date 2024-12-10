import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
import textwrap
from collections import defaultdict
import re
import json


matching = {
    "Conv": ["ConvInteger", "Cast", "Mul", "Relu", "DynamicQuantizeLinear"],
    "Gemm": ["DynamicQuantizeMatMul"],
    "MaxPool": ["MaxPool"],
    "GlobalAveragePool": ["GlobalAveragePool"],
    "ReorderOutput": ["ReorderOutput"],
    "Flatten": ["Flatten"],
    "Additional": ["ReorderInput", "Add"]
    }
    
def plot(orig_ops, quant_ops, output_name):
    # Prepare data for the first plot (Original Operations)
    orig_operations = [operation for operation in orig_ops]
    orig_durations = [orig_ops[operation]["duration"] for operation in orig_operations]

    # Plot the first horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(np.arange(len(orig_operations)), orig_durations, color="#1f77b4", label='Original')

    # Customize the first graph
    plt.title("ONNX-{model}-Original", loc='center')
    plt.xlabel('Average Duration - us')
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

        # Overlay red markers for original operator durations
        for op_idx, op_type in enumerate(matching_operations):
            if op_type in orig_ops:
                # Place a red marker above the total duration for this operation
                plt.plot(
                    orig_ops[op_type]["duration"],  # x-coordinate (duration)
                    op_idx,  # y-coordinate (operation index)
                    marker="o", color="red", markersize=8, label=None  # Red marker
                )

    # Customize the second graph
    plt.title("ONNX-{model}-Quantized", loc='center')
    plt.xlabel('Average Duration - us')
    plt.ylabel('Operation Types')
    plt.yticks(np.arange(n_operations), matching_operations)  # Only use matching keys for labels
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend below the plot

    # Save the second plot
    plt.tight_layout()
    plt.savefig(output_name + "_quantized.png")
    plt.close()


def consolidate_results(result_format, n):
    ops = defaultdict(lambda: {'duration': 0, 'count': 0})
    for i in range(n):
        with open(f'{result_format}_{i}.json', 'r') as f:
            data = json.load(f)
            
        filtered_data = [entry for entry in data if entry['cat']=='Node' and 'op_name' in entry['args'] and entry['dur']>0]

        for data in filtered_data:
            op_name = data['args']['op_name']
            ops[op_name]['duration'] += data['dur']
            ops[op_name]['count'] += 1

    for op in ops:
        print(f"Operator: {op}, Duration: {ops[op]['duration']}, Count: {ops[op]['count']}")
        ops[op]['duration'] /= n
        ops[op]['count'] /= n
        
    
    return ops

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model name', required=True)
    parser.add_argument('--orig_result_format', help='The path of the JSON file with the result from the original model', required=True)
    parser.add_argument('--quant_result_format', help='The path of the JSON file with the result from the quantized model', required=True)
    parser.add_argument('--num_repetitions', help='The number of time the profiler was run', required=True)
    parser.add_argument('--output_name', help='The name for the output plots', required=True)
    args = parser.parse_args()
    if args.model != "ResNet50":
        raise NotImplementedError("Currently, this code has not been extended for models other than ResNet50")
    print("Original Model:")
    orig_ops = consolidate_results(args.orig_result_format, int(args.num_repetitions))
    print("\nQuantized Model:")
    quant_ops = consolidate_results(args.quant_result_format, int(args.num_repetitions))
    plot(orig_ops, quant_ops, args.output_name)


