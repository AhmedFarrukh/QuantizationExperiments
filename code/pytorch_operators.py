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
                    'conv2d': 'conv2d',
                    'convolution': 'convolution',
                    '_convolution': '_convolution',
                    'mkldnn_convolution': 'mkldnn_convolution',
                    'empty': 'empty',  
                    'as_strided_': 'as_strided_',     
                    'resize_': 'resize_',       
                    'batch_norm': 'batch_norm',        
                    '_batch_norm_impl_index': '_batch_norm_impl_index',         
                    'native_batch_norm': 'native_batch_norm',
                    'empty_like': 'empty_like',      
                    'relu_': 'relu_',     
                    'clamp_min_': 'clamp_min_',   
                    'max_pool2d': 'max_pool2d',      
                    'max_pool2d_with_indices': 'max_pool2d_with_indices',        
                    'add_': 'add_',         
                    'adaptive_avg_pool2d': 'adaptive_avg_pool2d',        
                    'mean': 'mean',        
                    'sum': 'sum',         
                    'fill_': 'fill_',        
                    'div_': 'div_',      
                    'to': 'to',       
                    '_to_copy': '_to_copy',      
                    'empty_strided': 'empty_strided',       
                    'copy_': 'copy_',      
                    'flatten': 'flatten',        
                    'view': 'view',      
                    'linear + t + transpose + as_strided + addmm + expand + resolve_conj': 'quantized::linear_dynamic'    
                    }

    
def plot(orig_ops, quant_ops, output_name, model):
    if model == "ResNet50":
        matching = ResNet50_matching
    else:
        raise NotImplementedError
    # Prepare data for the first plot (Original Operations)
    orig_operations = [operation for operation in orig_ops]
    orig_durations = [orig_ops[operation]["duration"] for operation in orig_operations]

    # Plot the first horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(np.arange(len(orig_operations)), orig_durations, color="#1f77b4", label='Original')

    # Customize the first graph
    plt.title(f"PyTorch-{model}-Original", loc='center')
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

#linear + t + transpose + as_strided + addmm + expand + resolve_conj
    # Overlay red markers for original operator durations
    orig_ops_matching = copy.deepcopy(orig_ops)
    if model == "ResNet50":
        del orig_ops_matching["linear"]
        del orig_ops_matching["t"]
        del orig_ops_matching["transpose"]
        del orig_ops_matching["as_strided"]
        del orig_ops_matching["addmm"]
        del orig_ops_matching["expand"]
        del orig_ops_matching["resolve_conj"]
        orig_ops_matching["linear + t + transpose + as_strided + addmm + expand + resolve_conj"] = {"duration":0, "count":0}
        orig_ops_matching["linear + t + transpose + as_strided + addmm + expand + resolve_conj"]["duration"] = orig_ops["linear"]["duration"] + orig_ops["t"]["duration"] + orig_ops["transpose"]["duration"] + orig_ops["as_strided"]["duration"] + orig_ops["addmm"]["duration"] + orig_ops["expand"]["duration"] + orig_ops["resolve_conj"]["duration"]
        orig_ops_matching["linear + t + transpose + as_strided + addmm + expand + resolve_conj"]["count"] = orig_ops["linear"]["count"] + orig_ops["t"]["count"] + orig_ops["transpose"]["count"] + orig_ops["as_strided"]["count"] + orig_ops["addmm"]["count"] + orig_ops["expand"]["count"] + orig_ops["resolve_conj"]["count"] 


    for op_idx, op_type in enumerate(matching_operations):
        if op_type in orig_ops_matching:
            # Place a red marker above the total duration for this operation
            plt.plot(
                orig_ops_matching[op_type]["duration"],  # x-coordinate (duration)
                op_idx,  # y-coordinate (operation index)
                marker="o", color="red", markersize=8, label=None  # Red marker
            )

    # Customize the second graph
    plt.title(f"PyTorch-{model}-Quantized", loc='center')
    plt.xlabel('Average Duration - ms')
    plt.ylabel('Operation Types')
    plt.yticks(np.arange(n_operations), matching_operations)  # Only use matching keys for labels
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)  # Move legend below the plot

    # Save the second plot
    plt.tight_layout()
    plt.savefig(output_name + "_quantized.png", bbox_inches='tight')
    plt.close()


def parse_results(result_path):
    # Define markers for the section of interest
    table_start = r"-+\s+-+\s+-+\s+-+\s+-+\s+-+\s+-+"  # Matches the header divider
    table_end = r"Self CPU time total:"
    
    # Read the file content
    with open(result_path, 'r') as file:
        content = file.read()
    
    # Extract the block containing the table
    match = re.search(f"{table_start}(.*?){table_end}", content, re.DOTALL)
    if not match:
        raise ValueError("Table not found in the file")
    
    table_block = match.group(1).strip()
    
    # Parse the table into a dictionary
    lines = table_block.split("\n")
    headers = re.split(r"\s{2,}", lines[0].strip())  # Extract headers using consistent spacing
    data_lines = lines[2:]  # The rest are data rows
    
    # Create a dictionary for operations
    ops = {}
    for line in data_lines:
        # Skip lines that are empty or separators
        if not line.strip() or re.match(r"-+", line.strip()):
            continue
        
        # Split line into fields using consistent spacing
        fields = re.split(r"\s{2,}", line.strip())
        
        if len(fields) < len(headers):
            continue  # Skip lines that don't match the expected structure
        
        # Extract fields based on known header order
        op_name = fields[0].removeprefix("aten::")
        total_cpu_time = parse_time(fields[4])
        num_calls = int(fields[6])
        
        # Add to the dictionary
        ops[op_name] = {"count": num_calls, "duration": total_cpu_time}
    
    for op in ops:
        print(f"Operator: {op}, Duration: {ops[op]['duration']}, Count: {ops[op]['count']}")

    return ops

def parse_time(time_str):
    if time_str.endswith("ms"):
        return float(time_str[:-2])
    elif time_str.endswith("us"):
        return float(time_str[:-2]) / 1000.0  # Convert microseconds to milliseconds
    else:
        raise ValueError(f"Unknown time unit in: {time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model name', required=True)
    parser.add_argument('--orig_result_path', help='The path of the txt file with the result from the original model', required=True)
    parser.add_argument('--quant_result_path', help='The path of the txt file with the result from the quantized model', required=True)
    parser.add_argument('--output_name', help='The name for the output plots', required=True)
    args = parser.parse_args()
    if args.model not in ["ResNet50"]:
        raise NotImplementedError("Currently, this code has not been extended for models other than ResNet50.")
    print(f"Original Model - {args.model}:")
    orig_ops = parse_results(args.orig_result_path)
    print(f"Ouantized Model - {args.model}:")
    quant_ops = parse_results(args.quant_result_path)
    plot(orig_ops, quant_ops, args.output_name, args.model) 
