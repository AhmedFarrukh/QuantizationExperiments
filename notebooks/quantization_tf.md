:::{.cell}
## Dynamic Range Quantization in Tensorflow
We will now quantize the models in tensorflow. 
:::

:::{.cell}
### Loading the Code
First, let's get the clone the GitHub repository on the Chameleon server.
:::

:::{.cell .code}
```python
node.run('git clone https://github.com/AhmedFarrukh/QuantizationExperiments.git')
```
:::

:::{.cell}
### Install Python packages
Now, let's install the neccessary Python packages.
:::

:::{.cell .code}
```python
node.run('python3 -m pip install --user tensorflow==2.13.0')
```
:::

:::{.cell}
### Running the Python script 
We are now all set to run the Python script that will load models using `Keras`, and apply Dynamic Range Quantization. Both the original models and their quantized version will be saved in the `/home/cc/models` directory; you can change this directory by editing the command line arguments in the following cell. The models will be saved in `tflite` format.
:::

:::{.cell .code}
```python
node.run('python3 ./QuantizationExperiments/code/dynamicRangeQuantization_tf.py --dir=/home/cc/models')
```
:::
