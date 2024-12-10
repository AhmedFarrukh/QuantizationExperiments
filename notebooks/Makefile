# Define notebooks and their respective markdown files
NOTEBOOKS := quantization_TensorFlow.ipynb quantization_PyTorch.ipynb quantization_ONNX.ipynb onnxruntime_profiling.ipynb tflite_profiling.ipynb

# List of markdown files that form specific notebooks
quantization_TensorFlow_md := quantization_header.md reserve_top.md quantization_tf.md ssh_transfer.md reserve_end.md
quantization_PyTorch_md := quantization_header.md reserve_top.md quantization_pytorch.md ssh_transfer.md reserve_end.md
quantization_ONNX_md := quantization_header.md reserve_top.md quantization_onnx.md ssh_transfer.md reserve_end.md
onnxruntime_profiling_md := onnxruntime_profiler_header.md reserve_top.md onnxruntime_profiler.md reserve_end.md
tflite_profiling_md := tflite_profiler_header.md reserve_top.md tflite_profiler.md reserve_end.md

# Default target to build all notebooks
all: $(NOTEBOOKS)

# Clean target to remove generated notebooks
clean:
	rm -f $(NOTEBOOKS)

# Rule to create quantization_TensorFlow notebook by combining markdown files
quantization_TensorFlow.ipynb: $(quantization_TensorFlow_md)
	cat $(quantization_TensorFlow_md) | pandoc --wrap=none -o $@

quantization_PyTorch.ipynb: $(quantization_PyTorch_md)
	cat $(quantization_PyTorch_md) | pandoc --wrap=none -o $@

quantization_ONNX.ipynb: $(quantization_ONNX_md)
	cat $(quantization_ONNX_md) | pandoc --wrap=none -o $@

onnxruntime_profiling.ipynb: $(onnxruntime_profiling_md)
	cat $(onnxruntime_profiling_md) | pandoc --wrap=none -o $@

tflite_profiling.ipynb: $(tflite_profiling_md)
	cat $(tflite_profiling_md) | pandoc --wrap=none -o $@