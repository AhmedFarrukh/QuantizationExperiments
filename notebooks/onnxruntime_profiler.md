:::{.cell}
## Profiling Models
We will now use the ONNX runtime profiler to benchmark the performance of our models 
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
node.run('python3 -m pip install --user tf2onnx==1.16.1 onnxruntime==1.19.2 gdown==5.2.0 tensorflow==2.13.0 matplotlib==3.7.5')
node.run('export PATH=\"$PATH:/home/cc/.local/bin\"')
```
:::

:::{.cell}
### Loading the Models
The original and quantized versions of the models in our experiment are available on Google Drive, in both `.onnx` and `.tflite` format. We can load these model from the Drive.
:::

:::{.cell .code}
```python
node.run('gdown https://drive.google.com/drive/folders/1OcJ9ceYg6ZWFJ4QMR0zznsw0KVeHPa4h?usp=drive_link -O /home/cc/tflite_models --folder')
node.run('gdown https://drive.google.com/drive/folders/1YD2eW0557lorRmmP5izPiVf5anjdFgdc?usp=drive_link -O /home/cc/onnx_models --folder')
```
:::

:::{.cell}
We can now convert the `.tflite` models into `.onnx`.
:::

:::{.cell .code}
```python
node.run('python3 /home/cc/QuantizationExperiments/code/tflite_to_onnx.py --tflite_dir=/home/cc/tflite_models --onnx_dir=/home/cc/tflite_to_onnx_models')
```
:::

:::{.cell}
### Profiling ONNX Models 
Finally, we can run the profiler for both sets of models (ONNX and TFlite). For each model, the results from the profiler are saved in a JSON file. We then parse this JSON file and create plots of relevant results.
:::

:::{.cell .code}
```python
node.run('mkdir /home/cc/onnxruntime_profiling_results')
node.run('cd /home/cc/onnxruntime_profiling_results')
node.run('python3 /home/cc/QuantizationExperiments/code/onnx_profiling.py --tflite_dir=/home/cc/tflite_to_onnx_models --onnx_dir=/home/cc/onnx_models')
node.run('python3 /home/cc/QuantizationExperiments/code/plot_results.py --tflite_dir=. --onnx_dir=. --save_dir=/home/cc/plots')
```
:::

:::{.cell}
### Transfer Plots to Jupyter Interface 
Paste the output of the following cell in a terminal on your Jupyter Interface.
:::

:::{.cell .code}
```python
current_directory = os.getcwd()
print(f'scp -r cc@{reserved_fip}:/home/cc/plots {current_directory}')

```
:::

:::{.cell}
Finally, we can print the results.
:::

:::{.cell .code}
```python
import os
from IPython.display import Image, display
import glob

image_dir = current_directory + '/models' 
image_files = glob.glob(os.path.join(image_dir, '*.png'))

for image_file in image_files:
    display(Image(filename=image_file))

```
:::
