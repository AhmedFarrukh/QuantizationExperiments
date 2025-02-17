# Define notebooks and their respective markdown files
NOTEBOOKS := quantization_TensorFlow.ipynb quantization_PyTorch.ipynb quantization_ONNX.ipynb onnxruntime_profiling.ipynb tflite_profiling.ipynb pytorch_profiling.ipynb

# Directory for markdown files
MD_DIR := ../notebook_fragments

# List of markdown files that form specific notebooks
quantization_TensorFlow_md := $(MD_DIR)/quantization_header.md $(MD_DIR)/reserve_top.md $(MD_DIR)/quantization_tf.md $(MD_DIR)/ssh_transfer.md $(MD_DIR)/reserve_end.md
quantization_PyTorch_md := $(MD_DIR)/quantization_header.md $(MD_DIR)/reserve_top.md $(MD_DIR)/quantization_pytorch.md $(MD_DIR)/ssh_transfer.md $(MD_DIR)/reserve_end.md
quantization_ONNX_md := $(MD_DIR)/quantization_header.md $(MD_DIR)/reserve_top.md $(MD_DIR)/quantization_onnx.md $(MD_DIR)/ssh_transfer.md $(MD_DIR)/reserve_end.md
onnxruntime_profiling_md := $(MD_DIR)/onnxruntime_profiler_header.md $(MD_DIR)/reserve_top.md $(MD_DIR)/onnxruntime_profiler.md $(MD_DIR)/reserve_end.md
tflite_profiling_md := $(MD_DIR)/tflite_profiler_header.md $(MD_DIR)/reserve_top.md $(MD_DIR)/tflite_profiler.md $(MD_DIR)/reserve_end.md
pytorch_profiling_md := $(MD_DIR)/pytorch_profiler_header.md $(MD_DIR)/reserve_top.md $(MD_DIR)/pytorch_profiler.md $(MD_DIR)/reserve_end.md


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

pytorch_profiling.ipynb: $(pytorch_profiling_md)
	cat $(pytorch_profiling_md) | pandoc --wrap=none -o $@