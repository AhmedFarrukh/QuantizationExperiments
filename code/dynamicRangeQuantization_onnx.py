import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnx import hub
import pathlib
import argparse

modelNames = ["mobilenetv2-10", "inception_v2", "resnet50_v1", "resnet101_v1", "resnet152_v1", "vgg16", "vgg19"]

def quantize(save_dir):
    onnx_models_dir = pathlib.Path(save_dir)
    onnx_models_dir.mkdir(exist_ok=True, parents=True)

    for modelName in modelNames:
        model = hub.load(modelName)

        original_model_path = os.path.join(save_dir, f"{modelName}.onnx")
        quantized_model_path = os.path.join(save_dir, f"{modelName}_quant.onnx")

        onnx.save(model, original_model_path)

        # Perform dynamic range quantization
        quantize_dynamic(
            original_model_path,  # input model
            quantized_model_path,  # save quantized model
            weight_type=QuantType.QUInt8  # Quantize weights to 8 bits
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='The directory to save the models', required=True)
    
    args = parser.parse_args()

    quantize(args.dir)
