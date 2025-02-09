:::{.cell}
## Profiling Models
We will now use the TFlite benchmark to analyse the performance of our models 
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
node.run('python3 -m pip install --user gdown==5.2.0 matplotlib==3.7.5 pandas==2.0.3')
node.run('export PATH=\"$PATH:/home/cc/.local/bin\"')
```
:::

:::{.cell}
### Loading the Models
The original and quantized versions of the models in our experiment are available on Google Drive, in both `.tflite` format. We can load these model from the Drive.
:::

:::{.cell .code}
```python
node.run('/home/cc/.local/bin/gdown https://drive.google.com/drive/folders/1OcJ9ceYg6ZWFJ4QMR0zznsw0KVeHPa4h?usp=drive_link -O /home/cc/tflite_models --folder')
```
:::


:::{.cell}
### Profiling TFlite Models 
Finally, we can download and run the benchmark. We then parse the output and create plots of relevant results.
:::

:::{.cell .code}
```python
node.run('mkdir /home/cc/tflite_profiling_results')
node.run('python3 /home/cc/QuantizationExperiments/code/tflite_profiling.py  --tflite_dir=/home/cc/tflite_models --results_dir=/home/cc/tflite_profiling_results')
node.run('mkdir /home/cc/tflite_plots')
node.run('python3 /home/cc/QuantizationExperiments/code/tflite_plots.py --results_dir=/home/cc/tflite_profiling_results --save_dir=/home/cc/tflite_plots --num_repetitions=100')
node.run('python3 /home/cc/QuantizationExperiments/code/tflite_operators.py --model=ResNet50 --orig_result_path=/home/cc/tflite_profiling_results/tflite_ResNet50_profiling.txt --quant_result_path=/home/cc/tflite_profiling_results/tflite_ResNet50_quant_profiling.txt --output_name=/home/cc/tflite_plots/ResNet50')
node.run('python3 /home/cc/QuantizationExperiments/code/tflite_operators.py --model=VGG16 --orig_result_path=/home/cc/tflite_profiling_results/tflite_VGG16_profiling.txt --quant_result_path=/home/cc/tflite_profiling_results/tflite_VGG16_quant_profiling.txt --output_name=/home/cc/tflite_plots/VGG16')
node.run('python3 /home/cc/QuantizationExperiments/code/tflite_operators.py --model=MobileNetV2 --orig_result_path=/home/cc/tflite_profiling_results/tflite_MobileNetV2_profiling.txt --quant_result_path=/home/cc/tflite_profiling_results/tflite_MobileNetV2_quant_profiling.txt --output_name=/home/cc/tflite_plots/MobileNetV2')
```
:::

:::{.cell}
### Transfer Plots to Jupyter Interface 
Paste the output of the following cell in a terminal on your Jupyter Interface.
:::

:::{.cell .code}
```python
current_directory = os.getcwd()
!mkdir {NODE_TYPE}
print(f'scp -r cc@{reserved_fip}:/home/cc/tflite_plots {current_directory}/{NODE_TYPE}')

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

image_dir = current_directory + f'/{NODE_TYPE}/tflite_plots' 
image_files = glob.glob(os.path.join(image_dir, '*.png'))

for image_file in image_files:
    display(Image(filename=image_file))

```
:::
