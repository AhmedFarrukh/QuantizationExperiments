:::{.cell}
# ONNX Runtime Profililing
In this notebook, we use the ONNX runtime profiler to benchmark the performance of Quantized CNN models, in addition to their original versions. The models considered are: MobileNet, InceptionV3, ResNet50, ResNet101, ResNet152, VGG16, VGG19.

The models were loaded and quantized in ONNX. In addition to comparing the performance of models, an operator-level analysis is also conducted for ResNet50.
:::
