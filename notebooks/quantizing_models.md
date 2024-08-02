:::{.cell}
## Applying Quantization
:::

:::{.cell}
This notebook illustrates the process of applying Quantization to Deep Learning Models.
:::

:::{.cell}
Quantization, a technique that compresses models by converting parameters from floating point to fixed width numbers, can be useful to prepare models for deployment on edge devices. It significantly decreases the size of models and may improve inference times as well. In this notebook, [Post-training Dynamic Range Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization) is implemented, which converts model weights from floating point to integers with 8-bit precision.
:::

:::{.cell}
The technique is applied on popular Convolutional Neural Networks: MobileNet, InceptionV3, Resnet50, ResNet101, ResNet152, VGG16, VGG19.  
:::

:::{.cell}
First, the neccessary libraries are imported.
:::


:::{.cell .code}
```
import tensorflow as tf
import numpy as np
import os
import numpy as np
import pathlib
```
:::

:::{.cell}
A list of the names of the models to be quantized is defined.
:::


:::{.cell .code}
```
modelNames = ["MobileNet", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]
```
:::

:::{.cell}
The models are loaded using the Keras API. One instance of each model is converted to tflite format without any optimization; another instance is optimized using Dynamic Range Quantization.
:::

:::{.cell}
Both the original and the quantized versions of the models are stored in the `./tflite_models directory`.
:::


:::{.cell .code}
```
for modelName in modelNames:
  model_class = getattr(tf.keras.applications, modelName)
  model = model_class(weights='imagenet')

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model_quant = converter.convert()

  tflite_models_dir = pathlib.Path("/root/tflite_models/")
  tflite_models_dir.mkdir(exist_ok=True, parents=True)

  # Save the unquantized/float model:
  tflite_model_file = tflite_models_dir/(modelName+".tflite")
  tflite_model_file.write_bytes(tflite_model)
  # Save the quantized model:
  tflite_model_quant_file = tflite_models_dir/(modelName+"_quant.tflite")
  tflite_model_quant_file.write_bytes(tflite_model_quant)
```
:::

:::{.cell}
Next, we can verify that all models have been correctly saved by printing the contents of the `./tflite_models` directory. You should be able to see two files for each model, one for the original and one for the quantized version. Note that the size of the original models is bigger than the size of their quantized version.
:::


:::{.cell .code}
```
!ls -l ./tflite_models
```
:::
