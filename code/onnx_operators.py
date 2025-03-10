import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
import textwrap
from collections import defaultdict
import re
import json


matching = {
    "Conv": ["ConvInteger", "Cast", "Mul", "Relu", "Add", "DynamicQuantizeLinear", "ReorderInput"],
    "MaxPool": ["MaxPool"],
    "GlobalAveragePool": ["GlobalAveragePool"],
    "ReorderOutput": ["ReorderOutput"],
    "Flatten": ["Flatten"],
    "Gemm": ["DynamicQuantizeMatMul"]
    #"Additional": []
    }
    
def plot(orig_ops, quant_ops, output_name, model):
    # Prepare data for the first plot (Original Operations)
    orig_operations = [operation for operation in orig_ops]
    orig_durations = [orig_ops[operation]["duration"] for operation in orig_operations]

    # Plot the first horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(np.arange(len(orig_operations)), orig_durations, alpha = 0.8, label='Original')

    # Customize the first graph
    plt.title(f"ONNX-{model}-Original", loc='center')
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
                    alpha = 0.8,  
                    label=eq_op
                )
                stack_durations[op_idx] += quant_ops[eq_op]["duration"]  # Update baseline for this row
                quant_ops_plotted.add(eq_op)

        # Overlay red markers for original operator durations
        for op_idx, op_type in enumerate(matching_operations):
            if op_type in orig_ops:
                # Place a red marker above the total duration for this operation
                plt.plot(
                    orig_ops[op_type]["duration"],  # x-coordinate (duration)
                    op_idx,  # y-coordinate (operation index)
                    marker="o", color="red", markersize=8, label=None  # Red marker
                )
    if set(quant_ops) - quant_ops_plotted:
            print("ERROR: THE FOLLOWING OPERATIONS WERE NOT MAPPED!")
            print(set(quant_ops) - quant_ops_plotted)

    # Customize the second graph
    plt.title(f"ONNX-{model}-Quantized", loc='center')
    plt.xlabel('Average Duration - ms')
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
        ops[op]['duration'] /= n*1000
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
    for op in orig_ops:
        print(f"Operator: {op}, Duration: {orig_ops[op]['duration']}, Count: {orig_ops[op]['count']}")
    print("\nQuantized Model:")
    quant_ops = consolidate_results(args.quant_result_format, int(args.num_repetitions))
    for op in quant_ops:
        print(f"Operator: {op}, Duration: {quant_ops[op]['duration']}, Count: {quant_ops[op]['count']}")
    plot(orig_ops, quant_ops, args.output_name, args.model)


