#main dependency: TF version 2.17.0

import tensorflow as tf
import pathlib
import argparse

modelNames = ["MobileNet", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]

def quantize(save_dir):
    tflite_models_dir = pathlib.Path(save_dir)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    for modelName in modelNames:
        model_class = getattr(tf.keras.applications, modelName)
        model = model_class(weights='imagenet')

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model_quant = converter.convert()

        # Save the unquantized/float model:
        tflite_model_file = tflite_models_dir/(modelName+".tflite")
        tflite_model_file.write_bytes(tflite_model)
        # Save the quantized model:
        tflite_model_quant_file = tflite_models_dir/(modelName+"_quant.tflite")
        tflite_model_quant_file.write_bytes(tflite_model_quant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='The directory to save the models', required=True)
    
    args = parser.parse_args()

    quantize(args.dir)
