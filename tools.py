import torch
from typing import Callable, Dict, List
from torch import nn


def __verify_config(config: Dict, keys_to_verify: List):
    missing = []
    for k in keys_to_verify:
        if k not in config:
            missing.append(k)
    if missing:
        raise ValueError(f"The configuration needs to include: {missing}")


def torch2marabou_onnx(config: Dict, model_loader: Callable):
    __verify_config(config, ["input_shape", "out_file_path"])
    model: nn.Module = model_loader().cpu()
    model.eval()
    batch_size = 16
    torch_in = torch.randn(batch_size, config['input_shape'], requires_grad=True)
    torch.onnx.export(model,  # model being run
                      torch_in,  # model input (or a tuple for multiple inputs)
                      config['out_file_path'],  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})


def convert_h52onnx(model_path: str, arc: List, out_path: str):
    """
    A tool to convert h5 keras saved models used in the FGP paper to onnx, which can later be used with the Marabou
    framework.
    :param model_path: path to the .h5 file storing the model (models under models.fgp)
    :param arc: architecture of the network (by the file name, e.g mnist-20.20.20.h5 expects a list [20, 20, 20])
    :param out_path: the output saving path
    """
    from models.fgp.checkable_models import CheckableModel
    from tensorflow.keras import Model
    import keras2onnx
    checkable_model = CheckableModel((28 * 28,), arc, 10)
    original_model = checkable_model.model
    original_model.load_weights(model_path)
    # removing last layer as it is a softmax activation layer which isn't supported in onnx Marabou
    model2 = Model(inputs=original_model.input, outputs=original_model.layers[-2].output)
    onnx_model = keras2onnx.convert_keras(model2, model2.name)
    keras2onnx.save_model(onnx_model, out_path)
