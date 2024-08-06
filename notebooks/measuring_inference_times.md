:::{.cell}
## Measuring Inference Times
:::

:::{.cell}
In this notebook, we measure the inference times and memory footprint of 7 popular Convolutional Neural Network (CNN) models and their quantized versions. The CNN models are: MobileNet, InceptionV3, Resnet50, ResNet101, ResNet152, VGG16, VGG19.

The quantized models were created by applying [Post-training Dynamic Range Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization), which converts model weights from floating point numbers to 8-bit fixed width numbers.

Both the original models and their quantized versions are of tflite format, and were uploaded to [Google Drive](https://drive.google.com/drive/folders/1OcJ9ceYg6ZWFJ4QMR0zznsw0KVeHPa4h?usp=drive_link). 

The benchmarking of models is achieved by using the official [TFlite benchmark](https://www.tensorflow.org/lite/performance/measurement) which measures the following metrics:  
- Initialization time  
- Inference time of warmup state  
- Inference time of steady state   
- Memory usage during initialization time   
- Overall memory usage 

The benchmark generates a series of random inputs, runs the models and aggregates the results to report the aforementioned metrics.
:::

:::{.cell}
Before we begin, let's check the specifications of our hardware environment. The `lscpu` utility in Linux can be used to learn more about the CPU architecture. Amongst details to notice are the BogoMIPS value and clock speed which are both measurements of CPU speed; BogoMIPS ("*Bog*us" *M*illions of *I*nstructions per *S*econd) is calculated by the Linux kernel whereas clock speed is reported by the hardware manufacturer. Also pay attention to the flags, and see if there are any special deep learning optimizations, such as the AVX-512 VNNI isntruction set which accelerates convolutional neural networks.
:::
:::{.cell .code}
```python
!lscpu
```
:::

:::{.cell}
Now let's define the CNN models we will be using.
:::

:::{.cell .code}
```python
modelNames = ["MobileNet", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]
```
:::

:::{.cell}
We can download the models from the Google Drive using `gdown`. If you want to download your own set of models, you can modify the google drive link below. In this case, the `tflite_models` folder is downloaded from Google Drive and we will be able to access the models in the `./tflite_models` directory. 
:::

:::{.cell .code}
```python
import gdown
gdown.download_folder('https://drive.google.com/drive/folders/1OcJ9ceYg6ZWFJ4QMR0zznsw0KVeHPa4h')
```
:::

:::{.cell}
You can verify that the models were correctly loaded by listing the files in the `./tflite_models` directory. Note that there should be two `.tflite` files for each model: an original and a quantized version. The size of the quantized models should be about four times smaller than the size of the corresponding original model.
:::

:::{.cell .code}
```python
!ls -l ./tflite_models
```
:::

:::{.cell}
Next, we download the TFlite benchmark which we will use to measure inference times and memory footprint. More details about the benchmark can be found on the [tensorflow website](https://www.tensorflow.org/lite/performance/measurement). Note that the benchmark is specific to the architecture type (such as x86 or ARM), and the appropriate benchmark binary must be downloaded. Below, the benchmark is loaded for an x86-64 type architecture.

The benchmark is downloaded to the `./benchmark` directory, and its permissions are then updated to allow it to be executed.
:::


:::{.cell .code}
```python
!mkdir ./benchmark
!wget https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model -P ./benchmark
!chmod +x ./benchmark/linux_x86-64_benchmark_model
```
:::

:::{.cell}
Let's run the benchmark on the MobileNet_quant model and note the output.
:::

:::{.cell .code}
```python
!./benchmark/linux_x86-64_benchmark_model \
      --graph=./tflite_models/MobileNet_quant.tflite \
      --num_threads=1
```
:::

:::{.cell}
Let's define all the metrics that are reported by the benchmark:
:::

:::{.cell .code}
```pythonpython
metrics = ["Init Time (ms)", "Init Inference (ms)", "First Inference (ms)", "Warmup Inference (ms)", "Avg Inference (ms)", "Memory Init (MB)", "Memory Overall (MB)"]
```
:::

:::{.cell}
Since the result of the benchmark is reported as text on the console, we can define a parsing function to extract the data. The parsing function takes the output of the benchmark as an input and adds the results to a dictionary of metrics.

The function employs regular expressions to extract key performance metrics from the output logs. It defines specific patterns and attempts to match these against the output logs. When a match is identified, the corresponding metrics are stored in a dictionary provided to the function. The metrics, as defined earlier, serve as the keys in this dictionary. Each key is associated with an array that contains the values reported for that metric, allowing for organized and accessible data retrieval for further analysis.
:::


:::{.cell .code}
```python
import re

def parse_benchmark_output(output, results):
    """
    Parse benchmark output to extract model initialization times, inference timings, and memory footprint.
    """

    # Regular expressions to match the required information
    init_time_patterns = [
        re.compile(r'INFO: Initialized session in (\d+.\d+)ms.'),
        re.compile(r'INFO: Initialized session in (\d+)ms.')
    ]
    inference_patterns = [
        re.compile(r'INFO: Inference timings in us: Init: (\d+), First inference: (\d+), Warmup \(avg\): ([\d.e+]+), Inference \(avg\): ([\d.e+]+)'),
        re.compile(r'INFO: Inference timings in us: Init: (\d+), First inference: (\d+), Warmup \(avg\): ([\d.e+]+), Inference \(avg\): (\d+)'),
        re.compile(r'INFO: Inference timings in us: Init: (\d+), First inference: (\d+), Warmup \(avg\): (\d+.\d+), Inference \(avg\): (\d+.\d+)'),
        re.compile(r'INFO: Inference timings in us: Init: (\d+), First inference: (\d+), Warmup \(avg\): (\d+), Inference \(avg\): (\d+.\d+)'),
        re.compile(r'INFO: Inference timings in us: Init: (\d+), First inference: (\d+), Warmup \(avg\): (\d+), Inference \(avg\): (\d+)'),
    ]
    memory_patterns = [
        re.compile(r'INFO: Memory footprint delta from the start of the tool \(MB\): init=(\d+.\d+) overall=(\d+.\d+)'),
        re.compile(r'INFO: Memory footprint delta from the start of the tool \(MB\): init=(\d+.\d+) overall=(\d+)'),
        re.compile(r'INFO: Memory footprint delta from the start of the tool \(MB\): init=(\d+) overall=(\d+.\d+)'),
        re.compile(r'INFO: Memory footprint delta from the start of the tool \(MB\): init=(\d+) overall=(\d+)'),
    ]
    for line in output.split('\n'):
        # Match the initialization time
        for pattern in init_time_patterns:
            init_match = pattern.search(line)
            if init_match:
                results['Init Time (ms)'].append(float(init_match.group(1)))
                break

        # Match the inference timings
        for pattern in inference_patterns:
            inference_match = pattern.search(line)
            if inference_match:
                results["Init Inference (ms)"].append(int(inference_match.group(1))/1000)
                results["First Inference (ms)"].append(int(inference_match.group(2))/1000)
                results["Warmup Inference (ms)"].append(float(inference_match.group(3))/1000)
                results["Avg Inference (ms)"].append(float(inference_match.group(4))/1000)
                break

        # Match the memory footprint
        for pattern in memory_patterns:
            memory_match = pattern.search(line)
            if memory_match:
              results['Memory Init (MB)'].append(float(memory_match.group(1)))
              results['Memory Overall (MB)'].append(float(memory_match.group(2)))
              break

```
:::

:::{.cell}
Next, we can define a Pandas Dataframe to store our results. Since we will be repeatedly running the benchmark to estimate the standard deviation of results as well, for each metric we will define two columns - one for the mean and the other for the standard deviation.
:::

:::{.cell .code}
```python
import pandas as pd

# Define model types (rows)
rows = []
for model in modelNames:
  rows.append(model)
  rows.append(model + "_quant")

# Define columns
cols = []
for metric in metrics:
  cols.append(metric)
  cols.append(metric + "_sd")

# Create an empty DataFrame
finalResult = pd.DataFrame(index=rows, columns=cols)
```
:::

:::{.cell}
Finally, run the benchmark repeatedly and average the results. For each model, we repeatedly run the benchmark, and parse the output from the benchmark. After `n` trials, the mean and standard deviation of the metrics is added to the `finalResult` dataframe defined in the last step.
:::


:::{.cell .code}
```python
import subprocess
from collections import defaultdict
from statistics import mean
from statistics import stdev

n = 10 #the number of times the benchmark is called for each model

for modelName in rows:
  print(modelName)
  modelResults = defaultdict(list)
  for i in range(n):
    outputOriginal = subprocess.check_output("./benchmark/linux_x86-64_benchmark_model \
      --graph=./tflite_models/" + modelName +".tflite"+" \
      --num_threads=1", shell=True)
    outputOriginal = outputOriginal.decode('utf-8')
    output = parse_benchmark_output(outputOriginal, modelResults)

  for metric in metrics:
    finalResult.loc[modelName, metric] = mean(modelResults[metric])
    finalResult.loc[modelName, metric + "_sd"] = stdev(modelResults[metric])
```
:::

:::{.cell}
Let's have a look at the results.
:::

:::{.cell .code}
```python
print(finalResult)
```
:::

:::{.cell}
Let's create a directory to store the results from our experiment.
:::

:::{.cell .code}
```python
!mkdir ./results
```
:::

:::{.cell}
Let's convert the `finalResult` dataframe to a csv file and store it in the `./results` directory, allowing us to download that data for later use.
:::

:::{.cell .code}
```python
finalResult.to_csv("./results/finalResult.csv")
```
:::

:::{.cell}
Finally, we can generate plots of the results. We will display the bars for the original and quantized versions of each model side-by-side to facilitate easy comparison. Error bars, representing +/- one standard deviation, will also be included to provide an estimate of the variation.
:::


:::{.cell .code}
```python
import matplotlib.pyplot as plt
import numpy as np
for metric in metrics:
    means_orig = finalResult.loc[modelNames, metric].values
    errors_orig = finalResult.loc[modelNames, metric + "_sd"].values
    means_quant = finalResult.loc[[model + "_quant" for model in modelNames], metric].values
    errors_quant = finalResult.loc[[model + "_quant" for model in modelNames], metric + "_sd"].values


    n_groups = len(modelNames)
    index = np.arange(n_groups)

    fig, ax = plt.subplots()
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, means_orig, bar_width,
                     alpha=opacity,
                     yerr=errors_orig,
                     label='Original')

    rects2 = plt.bar(index + bar_width, means_quant, bar_width,
                     alpha=opacity,
                     yerr=errors_quant,
                     label='Quantized')

    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.title(f'Bar Chart for {metric}')
    plt.xticks(index + bar_width / 2, modelNames, rotation=45)
    plt.legend()

    plt.tight_layout()

    # Save the plot as an image
    plt.savefig("./results/" + metric + ".png")

    # Show the plot
    plt.show()
```
:::

:::{.cell}
As you look through the plots, pay particular attention to the Average Inference Time plot, and note if the quantization led to a decrease in inference time, and if so, by how much.

It is also interesting to note that sometimes even if the average inference time is greater for the quantized models, quantization might reduce other sources of latency, such as initialization time.
:::