<!-- Introduction - a couple of sentences about what we will do in this experiment -->

In the following experiments, we are going to measure the effect of quantization on deep learning models on different hardware environments. We will validate some results from the literature related to inference speedup due to quantization, and also resolve some discrepancies between different papers.

To run this experiment, ...

## Background

### Representing numbers

<!-- put some stuff on quantization here -->
While there are multiple ways to represent numbers in computing, Integers and Floating-Point numbers (also referred to as floats) are of particular relevance to machine learning. 

**Integers** are represented by a fixed number of bits, though the specific number of bits can vary across data-types. The number of bits used to store integers, as well as their signedness, determines the range of numbers that can be represented. Working with integers is relatively computationally inexpensive, quick and accurate. However, using a fixed number of bits limits the range of integers. In addition, they do not allow fractional precision.

**Floating-Point** numbers are represented in a different, though useful, format. The binary representation of a floating point number is split into three sections: sign, exponent and mantissa.
<table>
  <tr>
    <th>S</th>
    <th>EXPONENT</th>
    <th>MANTISSA</th>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

### Quantization and deep learning

<!-- start with reference to the survey paper by jiasi chen et al -->

<!-- then describe results from the two papers you looked at  - starting with fig 8 -->

## Run my experiment

### Create quantized models

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




