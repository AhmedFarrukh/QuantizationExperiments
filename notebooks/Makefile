# Define notebooks and their respective markdown files
NOTEBOOKS := quantization_TensorFlow.ipynb quantization_PyTorch.ipynb quantization_ONNX.ipynb

# List of markdown files that form specific notebooks
quantization_TensorFlow_md := quantization_header.md reserve_top.md quantization_tf.md ssh_transfer.md reserve_end.md
quantization_PyTorch_md := quantization_header.md reserve_top.md quantization_pytorch.md ssh_transfer.md reserve_end.md
quantization_ONNX_md := quantization_header.md reserve_top.md quantization_onnx.md ssh_transfer.md reserve_end.md

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