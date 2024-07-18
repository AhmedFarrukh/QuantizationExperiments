<!-- Introduction - a couple of sentences about what we will do in this experiment -->
In the following experiments, we are going to measure the effect of quantization on deep learning models on different hardware environments. We will validate some results from the literature related to inference speedup due to quantization, and also resolve some discrepancies between different papers.

<!--To run this experiment, ...-->

## Background

### Representing numbers
<!-- put some stuff on quantization here -->
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

The format of the numbers used to represent quantities associated with Deep Learning models (such as weights, activations and biases) can impact metrics such as accuracy, memory requirements, and inference time. 

### Quantization and Deep Learning
<!-- start with reference to the survey paper by jiasi chen et al -->
Deep Learning on Edge Devices, in contrast to a cloud-based approach, can decrease latency, offer better scalability and ensure greater privacy<sup>[1]</sup>. Deep Learning at the Edge has several applications including computer vision, natural language processing, and network functions. A familiar example of natural language processing on the edge are voice assistants, such as Amazon's Alexa and Apple's Siri, which use on-device processing to detect wakewords<sup>[1]</sup>. 

However, the resource-constrained environments of edge devices can present a challenge. Model Compression techniques can be used to prepare Deep Neural Networks for deployment on the edge, with minimal loss in accuracy<sup>[1]</sup>. Amongst such methods is Parameter Quantization which converts the parameters of an existing DNN from floating-point numbers to low bit-width numbers to avoid costly floating-point operations<sup>[1]</sup>. 
<!-- then describe results from the two papers you looked at  - starting with fig 8 -->
However, past literature reports varying results regarding the impact of Quantization. Q. Qin et al. reported that after quantizing the weights of populer Convolutional Neural Networks, their inference times were generally higher than the original models, though there was a decrease in memory footprint<sup>[2]</sup>. Krishnamoorthi also reports decrease in memory footprint, but reports a decrease in inference times post-quantization<sup>[3]</sup>. The tensorflow website also claims that quantization can lead to over 2x speedup<sup>[4]</sup>.

The discrepancy in results can be explained by the use of different quantization methods, frameworks and hardware environments. Nevertheless, the lack of unanimity in results make it difficult to make sound predictions and identify appropriate use cases. 

## Run my experiment

In our experiments, we use Tensorflow to apply [Post-training Dynamic Range Quantization](https://www.tensorflow.org/lite/performance/post_training_quant) on 7 popular Convolutional Neural Networks, and measure the inference times in different hardware environments.

### Create quantized models
<a target="_blank" href="https://colab.research.google.com/github/https://colab.research.google.com/github/AhmedFarrukh/DeepLearning-EdgeComputing/blob/main/notebooks/QuantizingModels.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Follow along this notebook to load CNN models, and apply Post-training Dynamic Range Quantization.
<!-- save in models subdirectory in this repo -->

### Measure inference time on Google Colab hosted runtime

### Measure inference time on cloud CPU

### Measure inference time on NVIDIA Jetson device

### Measure inference time on Raspberry Pi

### Analyze results

## Notes

The [PreliminaryExperiment_ResNet101V2](https://github.com/AhmedFarrukh/DeepLearning-EdgeComputing/blob/main/PreliminaryExperiment_ResNet101V2.ipynb) notebook contains a preliminary experiment, implementing post-training dynamic range quantization on a pre-trained ResNet101V2 model.

The [Copy of post_training_quant](https://github.com/AhmedFarrukh/DeepLearning-EdgeComputing/blob/main/Copy_of_post_training_quant.ipynb) and [Copy of post_training_integer_quant](https://github.com/AhmedFarrukh/DeepLearning-EdgeComputing/blob/main/Copy_of_post_training_integer_quant.ipynb) notebooks contains modified versions of official TF examples. 


### References
[1] J. Chen and X. Ran, "Deep Learning With Edge Computing: A Review," in Proceedings of the IEEE, vol. 107, no. 8, pp. 1655-1674, Aug. 2019, doi: 10.1109/JPROC.2019.2921977. 

[2] Q. Qin et al., "To Compress, or Not to Compress: Characterizing Deep Learning Model Compression for Embedded Inference," 2018 IEEE Intl Conf on Parallel & Distributed Processing with Applications, Ubiquitous Computing & Communications, Big Data & Cloud Computing, Social Computing & Networking, Sustainable Computing & Communications (ISPA/IUCC/BDCloud/SocialCom/SustainCom), Melbourne, VIC, Australia, 2018, pp. 729-736, doi: 10.1109/BDCloud.2018.00110.

[3] R. Krishnamoorthi, “Quantizing deep convolutional networks for efficient inference: A whitepaper,” arXiv:1806.08342 [cs, stat], Jun. 2018, Available: https://arxiv.org/abs/1806.08342

[4] https://www.tensorflow.org/lite/performance/post_training_quantization




