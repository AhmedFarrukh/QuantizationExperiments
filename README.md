# Benchmarking Inference Speedup due to Quantization
In our experiments, we tested the effect of Post-Training Dynamic Range Quantization across different models, frameworks and hardware. The experiments were conducted using [Chameleon](https://chameleoncloud.org/), an open testbed that provides access to different hardware environments. 

This repository contains notebooks for two main tasks: model quantization and inference time measurement. For each of these tasks, we provide notebooks implemented in three popular machine learning frameworks: ONNX, TFLite, and PyTorch. On PyTorch, convolution operators could not be quantized using Dynamic Range Quantization, so the results were not included in the findings. To test on different hardware, change the `NODE_TYPE` in the notebook. 

## Run the experiments
To get started, open the Jupyter Interface in Chameleon, initiate a terminal and run the following in the terminal:
```
cd ~/work
git clone https://github.com/AhmedFarrukh/QuantizationExperiments.git
```
Then open the `QuantizationExperiments/notebooks` directory and follow along the instructions in the notebooks.

## Reproduce plots
To reproduce the plots, you can run the following commands using the sample results available in the `results` directory. To reproduce the plots with new results, replace the results' directories in the following commands. 

```
python code/result1_frameworks.py --model ResNet50 --tflite_result=results/compute_cascadelake/tflite_profiling_results --onnx_result=results/compute_cascadelake/onnxruntime_profiling_results --num_repetitions=10 --output=.
```

```
python code/result2_hardware.py --model ResNet50 --tflite_result=results --num_repetitions=10 --output=.  
```

```
python code/result3_models.py --tflite_result=results/compute_cascadelake/tflite_profiling_results --onnx_result=results/compute_cascadelake/onnxruntime_profiling_results --num_repetitions=10 --output=.
```





