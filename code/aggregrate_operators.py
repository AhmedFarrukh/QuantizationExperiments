def aggregate_convolution(framework, orig_ops, quant_ops, model):
    
    if framework not in ['onnx', 'tflite']:
        raise NotImplemented

    conv_operators = dict()

    if model == "ResNet50":
        if framework == 'onnx':
            conv_operators['Convolution'] = orig_ops["Conv"]["duration"]
            conv_operators['Quantized Convolution'] = sum([quant_ops[op]["duration"] for op in ["ConvInteger", "Mul", "Relu", "Add"] if op in quant_ops])
            conv_operators['Convert'] = sum([quant_ops[op]["duration"] for op in ["Cast", "DynamicQuantizeLinear", "ReorderInput"] if op in quant_ops])

        elif framework == 'tflite':
            conv_operators['Convolution'] = sum([orig_ops[op]["duration"] for op in ["Convolution (NHWC, F32) IGEMM", "Convolution (NHWC, F32) GEMM"] if op in orig_ops])
            conv_operators['Quantized Convolution'] = sum([quant_ops[op]["duration"] for op in ["Convolution (NHWC, QDU8, F32, QC8W) IGEMM", "Convolution (NHWC, QD8, F32, QC8W) IGEMM"] if op in quant_ops])
            conv_operators['Convert'] = sum([quant_ops[op]["duration"] for op in ["Convert (NC, F32, QDU8)", "Convert (NC, F32, QD8)"] if op in quant_ops])

    if model == "MobileNetV2":
        if framework == 'onnx':
            conv_operators['Convolution'] = orig_ops["Conv"]["duration"]
            conv_operators['Quantized Convolution'] = sum([quant_ops[op]["duration"] for op in ["ConvInteger", "Mul", "Add"] if op in quant_ops])
            conv_operators['Convert'] = sum([quant_ops[op]["duration"] for op in ["Cast", "DynamicQuantizeLinear", "Clip"] if op in quant_ops])


        elif framework == 'tflite':
            conv_operators['Depthwise Convolution'] = orig_ops["Convolution (NHWC, F32) DWConv"]["duration"]
            conv_operators['Convolution'] = sum([orig_ops[op]["duration"] for op in ["Convolution (NHWC, F32) GEMM", "Convolution (NHWC, F32) IGEMM"] if op in orig_ops])
            conv_operators['Quantized Convolution'] = sum([quant_ops[op]["duration"] for op in ["Convolution (NHWC, QDU8, F32, QC8W) IGEMM", "Convolution (NHWC, QD8, F32, QC8W) IGEMM", "Convolution (NHWC, F32) GEMM", "Convolution (NHWC, F32) IGEMM"] if op in quant_ops])
            conv_operators['Quantized Depthwise Convolution'] = sum([quant_ops[op]["duration"] for op in ["Convolution (NHWC, QDU8, F32, QC8W) IGEMM", "Convolution (NHWC, QD8, F32, QC8W) IGEMM", "Convolution (NHWC, F32) GEMM", "Convolution (NHWC, F32) IGEMM"] if op in quant_ops])
            conv_operators['Convert'] = sum([quant_ops[op]["duration"] for op in ["Convert (NC, F32, QDU8)", "Convert (NC, F32, QD8)"] if op in quant_ops])
            conv_operators['Pad'] = quant_ops["Constant Pad (ND, X32)"]["duration"]

    if model == "VGG16":
        if framework == 'onnx':
            conv_operators['Convolution'] = orig_ops["Conv"]["duration"]
            conv_operators['Quantized Convolution'] = sum([quant_ops[op]["duration"] for op in ["ConvInteger", "Mul", "Relu", "Add"] if op in quant_ops])
            conv_operators['Convert'] = sum([quant_ops[op]["duration"] for op in ["Cast", "DynamicQuantizeLinear", "ReorderInput"] if op in quant_ops])

        elif framework == 'tflite':
            conv_operators['Convolution'] = orig_ops["Convolution (NHWC, F32) IGEMM"]["duration"]
            conv_operators['Quantized Convolution'] = sum([quant_ops[op]["duration"] for op in ["Convolution (NHWC, QDU8, F32, QC8W) IGEMM", "Convolution (NHWC, QD8, F32, QC8W) IGEMM"] if op in quant_ops])
            conv_operators['Convert'] = sum([quant_ops[op]["duration"] for op in ["Convert (NC, F32, QDU8)", "Convert (NC, F32, QD8)"] if op in quant_ops])
            
    return conv_operators

def aggregate_fullyconnected(framework, orig_ops, quant_ops, model):
    
    if framework not in ['onnx', 'tflite']:
        raise NotImplemented

    conv_operators = dict()

    if framework== 'onnx':
        conv_operators['Fully Connected'] = sum([quant_ops[op]["duration"] for op in ["Flatten", "GEMM"] if op in quant_ops])
        conv_operators['Quantized Fully Connected'] = sum([quant_ops[op]["duration"] for op in ["Flatten", "DynamicQuantizeMatMul"] if op in quant_ops])
        
    if framework == 'tflite':
        conv_operators['Fully Connected'] = orig_ops['Fully Connected (NC, F32) GEMM']["duration"]
        conv_operators['Quantized Fully Connected'] = sum([quant_ops[op]["duration"] for op in ["Fully Connected (NC, QDU8, F32, QC8W) GEMM", "Fully Connected (NC, QD8, F32, QC8W) GEMM"] if op in quant_ops])

    return conv_operators