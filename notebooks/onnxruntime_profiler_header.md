:::{.cell}
# ONNX Runtime Profililing
In this notebook, we use the ONNX runtime profiler to benchmark the performance of Quantized CNN models, in addition to their original versions. The models considered are: MobileNet, InceptionV3, ResNet50, ResNet101, ResNet152, VGG16, VGG19.

In order to compare the effect of the training framework, models quantized in both ONNX and Tensorflow are considered. One set of models was loaded, quantized and stored all in ONNX. The other set was loaded in TensorFlow, converted to TFlite and then stored in tflite format; before the profiler is used to benchmark these models, they are converted into ONNX.
:::
