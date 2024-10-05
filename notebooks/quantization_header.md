:::{.cell}
# Dynamic Range Quantization
In this notebook, we will apply Dynamic Range Quantization to a series of CNN models.

Dynamic Range Quantization converts model weights from computationally expensive floating point number to fixed-width numbers; the range for each activation is computed on runtime. In this notebook, we specifically implement quantization from float32 to 8-bit Ints. 

The models used are: MobileNet, InceptionV3, ResNet50, ResNet101, ResNet152, VGG16, VGG19.
:::
