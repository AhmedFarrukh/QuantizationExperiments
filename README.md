<!-- Introduction - a couple of sentences about what we will do in this experiment -->

In this experiment, we are going to measure inference time of some deep learning models. We will validate some results from the literature related to inference speedup due to quantization, and also resolve some discrepancies between different papers.

To run this experiment, ...

## Background

### Representing numbers

<!-- put some stuff on quantization here -->

### Quantization and deep learning

<!-- start with reference to the survey paper by jiasi chen et al -->

<!-- then describe results from the two papers you looked at  - starting with fig 8 -->

## Run my experiment

### Create quantized models

<!-- save in models subdirectory in this repo -->

### Measure inference time on Google Colab hosted runtime

### Measure inference time on cloud CPU

### Measure inference time on edge device (Raspberry Pi)

### Analyze results

## Notes

The [PreliminaryExperiment_ResNet101V2](https://github.com/AhmedFarrukh/DeepLearning-EdgeComputing/blob/main/PreliminaryExperiment_ResNet101V2.ipynb) notebook contains a preliminary experiment, implementing post-training dynamic range quantization on a pre-trained ResNet101V2 model.

The [Copy of post_training_quant](https://github.com/AhmedFarrukh/DeepLearning-EdgeComputing/blob/main/Copy_of_post_training_quant.ipynb) and [Copy of post_training_integer_quant](https://github.com/AhmedFarrukh/DeepLearning-EdgeComputing/blob/main/Copy_of_post_training_integer_quant.ipynb) notebooks contains modified versions of official TF examples. 


### References




