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
node.run('python3 -m pip install --user onnxruntime==1.19.2 gdown==5.2.0 matplotlib==3.7.5')
node.run('export PATH=\"$PATH:/home/cc/.local/bin\"')
```
:::

:::{.cell}
### Loading the Models
The original and quantized versions of the models in our experiment are available on Google Drive, in both `.onnx` format. We can load these model from the Drive.
:::

:::{.cell .code}
```python
node.run('/home/cc/.local/bin/gdown https://drive.google.com/drive/folders/1YD2eW0557lorRmmP5izPiVf5anjdFgdc?usp=drive_link -O /home/cc/onnx_models --folder')
```
:::


:::{.cell}
### Profiling ONNX Models 
Finally, we can run the profiler. For each model, the results from the profiler are saved in JSON files. We then parse this JSON files and create plots of relevant results.
:::

:::{.cell .code}
```python
node.run('mkdir /home/cc/onnxruntime_profiling_results')
node.run('python3 /home/cc/QuantizationExperiments/code/onnx_profiling.py  --onnx_dir=/home/cc/onnx_models --results_dir=/home/cc/onnxruntime_profiling_results --num_repetitions=10')
node.run('mkdir /home/cc/plots')
node.run('python3 /home/cc/QuantizationExperiments/code/onnx_plots.py --onnx_dir=/home/cc/onnxruntime_profiling_results --save_dir=/home/cc/plots --num_repetitions=10')
node.run('python3 /home/cc/QuantizationExperiments/code/onnx_operators.py --model=ResNet50 --orig_result_format=/home/cc/onnxruntime_profiling_results/onnx_ResNet50_profiling --quant_result_format=/home/cc/onnxruntime_profiling_results/onnx_ResNet50_quant_profiling --num_repetitions=10 --output_name=/home/cc/plots/ResNet50_OperatorLevel')
```
:::

:::{.cell}
### Transfer Plots to Jupyter Interface 
Paste the output of the following cell in a terminal on your Jupyter Interface.
:::

:::{.cell .code}
```python
current_directory = os.getcwd()
print(f'scp cc@{reserved_fip}:/home/cc/plots/* {current_directory}/{NODE_TYPE}')

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

image_dir = current_directory + f'/{NODE_TYPE}' 
image_files = glob.glob(os.path.join(image_dir, '*.png'))

for image_file in image_files:
    display(Image(filename=image_file))

```
:::
