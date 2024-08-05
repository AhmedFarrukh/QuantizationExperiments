<!-- Introduction - a couple of sentences about what we will do in this experiment -->

Quantization, a model compression technique, prepares Deep Learning models for deployment on edge devices. By converting model parameters to a lower precision representation, quantization reduces the memory footprint and can potentially reduce inference time.

However, in past literature, there is a discrepancy about the effect of quantization on inference time, with some papers reporting an increase while others suggest a decrease.

In the paper "To Compress, or Not to Compress: Characterizing Deep Learning Model Compression for Embedded Inference," Qin reported that quantization increased inference times of Convolutional Neural Networks<sup>[1]</sup>. The following bar chart from the paper represents how much the inference times increased post-quantization:

<img src="https://github.com/user-attachments/assets/3c5b11a8-d72b-4df1-b3dc-f3f86e35e197" width="600" height="300">

In our experiments, we aim to reproduce this plot and benchmark the inference time speed-up due to quantization on multiple hardware environments.


## Background

### Representing numbers
While there are multiple ways to represent numbers in computing, Integers and Floating-Point numbers (also referred to as floats) are of particular relevance to machine learning. 

**Integers** are represented by a fixed number of bits, though the specific number of bits can vary across data-types. The number of bits used to store integers, as well as their signedness, determines the range of numbers that can be represented. Working with integers is relatively computationally inexpensive, quick and accurate. However, using a fixed number of bits limits the range of integers. In addition, they do not allow fractional precision.

**Floating-Point** numbers are represented in a different format. The binary representation of a floating point number is split into three sections: sign, exponent and mantissa. The precise split depends on the standard used. For instance, according to the Single-Precision IEEE Floating Point Format, 1 bit is reserved for the sign, 8 bits for the exponent and the remaining 23 bits for the mantissa. 
<table><thead>
  <tr>
    <th>Sign</th>
    <th colspan="8">Exponent</th>
    <th colspan="23">Mantessa</th>
  </tr></thead>
<tbody>
  <tr>
    <td>31</td>
    <td>30</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>23</td>
    <td>22</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>0</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</tbody></table>
In order to understand how a decimal number can be represented in floating point format, it helps to recognize that binary numbers can have ‘decimal’ points and can be represented using scientific notation. For example, the decimal number 43.625 can be represented as (1.01011101)<sub>2</sub> × 2<sup>5</sup>. All binary numbers can be represented as the product of a significand (with a value between 1 and 2) and a power of 2. The sign bit in the floating point number represents the sign (and is 1 for negative numbers), the exponent stores the power of 2 (with a bias of 127 i.e. 127 is added to the power) and mantissa stores the binary digits after the decimal point.

An alternate way to consider floating point numbers is to think of the exponent as representing a window of numbers, and the mantissa acting as an offset within this window. For example, when the exponent represents a power of 2, the window is [4, 8). Other examples of windows are [8, 16), [16, 32), [32, 64) and so on. Note that as the exponent increases, the width of the window becomes larger. The window is divided into 2<sup>23</sup> increments (as the mantissa has 23 bits), and the specific value of the mantissa identifies the number to be represented. The width of the window varies but the number of increments in the window do not, and as a result, the precision of floating point numbers depends partly on the magnitude of the number itself.

Floating Point numbers represent a large range of numbers with comparatively few bits, and can represent decimal numbers. However, compared to integer numbers, they have a higher computation cost. 

The format of the numbers used to represent the parameters of Deep Learning models (such as weights, activations and biases) can impact metrics such as accuracy, memory requirements, and inference time. 

### Quantization and Deep Learning
<!-- start with reference to the survey paper by jiasi chen et al -->
Deep Learning on Edge Devices, in contrast to a cloud-based approach, can decrease latency, offer better scalability and ensure greater privacy<sup>[2]</sup>. Deep Learning at the Edge has several applications including computer vision, natural language processing, and network functions. A familiar example of natural language processing on the edge are voice assistants, such as Amazon's Alexa and Apple's Siri, which use on-device processing to detect wakewords<sup>[2]</sup>. 

However, the resource-constraints of edge devices can present a challenge. Model Compression techniques can be used to prepare Deep Neural Networks (DNNs) for deployment on the edge, with minimal loss in accuracy<sup>[2]</sup>. Amongst such methods is Parameter Quantization which converts the parameters of an existing DNN from floating-point numbers to low bit-width numbers to avoid costly floating-point operations<sup>[2]</sup>. 
<!-- then describe results from the two papers you looked at  - starting with fig 8 -->
However, past literature reports varying results regarding the impact of Quantization. Q. Qin et al. reported that after quantizing the weights of populer Convolutional Neural Networks, their inference times were generally higher than the original models, though there was a decrease in memory footprint<sup>[1]</sup>. Krishnamoorthi also reports decrease in memory footprint, but reports a decrease in inference times post-quantization<sup>[3]</sup>. The tensorflow website also claims that quantization can lead to over 2x speedup<sup>[4]</sup>.

The discrepancy in results can be explained by the use of different quantization methods, frameworks and hardware environments; in "Performance evaluation of INT8 quantized inference on mobile GPUs", inference time speedup is shown to vary depending on aforemention factors, supporting this explanation<sup>[5]</sup>.  

Nevertheless, the lack of unanimity in results make it difficult to make sound predictions and identify appropriate use cases. 

## Methodology
### Models
In our experiments, based on Kim's plot (presented earlier)<sup>[1]</sup>, we test on 7 popular Convolutional Neural Networks: MobileNet, InceptionV3, ResNet50, ResNet101, ResNet152, VGG16, VGG19. 

The quantized versions were generated by applying Tensorflow's [Post-training Dynamic Range Quantization](https://www.tensorflow.org/lite/performance/post_training_quant) to the original models. Post-training Dynamic Range Quantization converts the model weights to 8-bit fixed width numbers. The following notebook demonstrates how these models were quantized: 
<a target="_blank" href="https://colab.research.google.com/github/AhmedFarrukh/DeepLearning-EdgeComputing/blob/main/notebooks/quantizing_models.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Version</th>
      <th>Original Size (MB)</th>
      <th>Quantized Size (MB)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MobileNet</td>
      <td>V1</td>
      <td>17</td>
      <td>5</td>
    </tr>
    <tr>
      <td>Inception</td>
      <td>V3</td>
      <td>91</td>
      <td>24</td>
    </tr>
    <tr>
      <td>ResNet50</td>
      <td>V1</td>
      <td>98</td>
      <td>25</td>
    </tr>
    <tr>
      <td>ResNet101</td>
      <td>V1</td>
      <td>170</td>
      <td>44</td>
    </tr>
    <tr>
      <td>ResNet152</td>
      <td>V1</td>
      <td>230</td>
      <td>59</td>
    </tr>
    <tr>
      <td>VGG16</td>
      <td>V1</td>
      <td>528</td>
      <td>133</td>
    </tr>
    <tr>
      <td>VGG19</td>
      <td>V1</td>
      <td>549</td>
      <td>138</td>
    </tr>
  </tbody>
</table>

All models, both original and quantized, are of `.tflite` format and are available on [Google Drive](https://drive.google.com/drive/folders/1OcJ9ceYg6ZWFJ4QMR0zznsw0KVeHPa4h?usp=drive_link). 

### Benchmarking
The official [TFlite Benchmark](https://www.tensorflow.org/lite/performance/measurement) was used to measure performance metrics. The benchmark generates randon inputs, repeatedly runs the model and reports the aggregate latency and memory statistics. It reports the following output:  
- Initialization time  
- Inference time of warmup state  
- Inference time of steady state  
- Memory usage during initialization time  
- Overall memory usage

The benchmark is specific to the hardware type (such as x86_64/ARM64) and operating system, so it's important to ensure that the correct version of the benchmark is being used.

### Hardware Environments
We tested on the following hardware environments which, with the exception of Google Colab, are all available through Chameleon. 
<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Type</th>
      <th>Year of Release</th>
      <th>Deep Learning Optimization</th>
      <th>BogoMIPS</th>
      <th>Clock Speed (GHz)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Intel Ice Lake CPU</td>
      <td>Bare Metal</td>
      <td>2019</td>
      <td>AVX-512 VNNI</td>
      <td>4600.00</td>
      <td>2.30</td>
    </tr>
    <tr>
      <td>Intel Cascade Lake CPU</td>
      <td>Bare Metal</td>
      <td>2019</td>
      <td>AVX-512 VNNI</td>
      <td>4988.36</td>
      <td>2.80</td>
    </tr>
    <tr>
      <td>Intel Skylake CPU</td>
      <td>Bare Metal</td>
      <td>2015</td>
      <td>-</td>
      <td>5200.00</td>
      <td>3.00</td>
    </tr>
    <tr>
      <td>Intel Broadwell CPU</td>
      <td>Bare Metal</td>
      <td>2014</td>
      <td>-</td>
      <td>4000.24</td>
      <td>2.00</td>
    </tr>
    <tr>
      <td>Intel Haswell CPU</td>
      <td>Bare Metal</td>
      <td>2013</td>
      <td>-</td>
      <td>4599.98</td>
      <td>2.30</td>
    </tr>
    <tr>
      <td>Raspberry Pi 5 (ARM Cortex A76)</td>
      <td>Bare Metal</td>
      <td>2023</td>
      <td>Optimized integer and vector operations</td>
      <td>108.00</td>
      <td>2.40</td>
    </tr>
    <tr>
      <td>Raspberry Pi 4 (ARM Cortex A72)</td>
      <td>Bare Metal</td>
      <td>2019</td>
      <td>-</td>
      <td>108.00</td>
      <td>1.50</td>
    </tr>
    <tr>
      <td>Google Colab CPU Runtime</td>
      <td>Shared</td>
      <td>-</td>
      <td>-</td>
      <td>4399.99</td>
      <td>2.20</td>
    </tr>
  </tbody>
</table>

Some of the hardware, as shown, have special deep learning optimizations. The AVX-512 VNNI instruction set on newer Intel CPUs is designed to accelerate Convolutional Neural Networks. The new Raspberry Pi 5 also has special machine learning optimizations. 

BogoMips values, also reported in the table, are a measurement of CPU speed made by the Linux kernel.

## Run my experiment
In the following experiments, we first load the original and quantized models from Google Drive, then repeatedly run the benchmark on each model, and finally plot the results.

### Measure inference time on Google Colab hosted runtime
Click the button below to open the measuring_inference_times.ipynb notebook on Google Colab:

<a target="_blank" href="https://colab.research.google.com/github/AhmedFarrukh/DeepLearning-EdgeComputing/blob/main/notebooks/measuring_inference_times.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Measure inference time on a CPU (through Chameleon)
First, you'll run the `reserve.ipynb` notebook to reserve a resource on Chameleon and configure it with the software needed to run this experiment. At the end of this notebook, you'll set up an SSH tunnel between your local device and a Jupyter notebook server that you just created on your Chameleon resource. Then, you'll open the notebook server in your local browser and run the sequence of notebooks you see there.

To get started, open the Jupyter Interface in Chameleon, initiate a terminal and run the following in the terminal:
```
cd ~/work
git clone https://github.com/AhmedFarrukh/DeepLearning-EdgeComputing.git
```
Then open the `reserve.ipynb` notebook inside the `DeepLearning-EdgeComputing` directory and follow along the instructions there.


### Measure inference time on a Raspberry Pi (through Chameleon)

## Analyze results
Across the different hardware environments we tested on, we generally found quantization to be effective in reducing inference time, and the inference time of quantized models was observed to be lower than the inference time of original models. One notable exception was the Intel Broadwell CPU where the inference time increased after quantization. 

Quantization was more effective on newer Intel CPU microarchitectures that implemented the AVS-512 VNNI instruction set, and the decrease in inference times post-quantization was greater than older CPU microarchitectures.

A greater decrease in inference times post-quantization was observed on the Raspberry Pi 5, compared to the older Raspberry Pi 4. In general, inference on the Raspberry Pi 5 was about three times faster than Raspberry Pi 4.

## Notes



### References
[1] Q. Qin et al., "To Compress, or Not to Compress: Characterizing Deep Learning Model Compression for Embedded Inference," 2018 IEEE Intl Conf on Parallel & Distributed Processing with Applications, Ubiquitous Computing & Communications, Big Data & Cloud Computing, Social Computing & Networking, Sustainable Computing & Communications (ISPA/IUCC/BDCloud/SocialCom/SustainCom), Melbourne, VIC, Australia, 2018, pp. 729-736, doi: 10.1109/BDCloud.2018.00110.

[2] J. Chen and X. Ran, "Deep Learning With Edge Computing: A Review," in Proceedings of the IEEE, vol. 107, no. 8, pp. 1655-1674, Aug. 2019, doi: 10.1109/JPROC.2019.2921977. 

[3] R. Krishnamoorthi, “Quantizing deep convolutional networks for efficient inference: A whitepaper,” arXiv:1806.08342 [cs, stat], Jun. 2018, Available: https://arxiv.org/abs/1806.08342

[4] https://www.tensorflow.org/lite/performance/post_training_quantization

[5] S. Kim, G. Park and Y. Yi, "Performance Evaluation of INT8 Quantized Inference on Mobile GPUs," in IEEE Access, vol. 9, pp. 164245-164255, 2021, doi: 10.1109/ACCESS.2021.3133100.




