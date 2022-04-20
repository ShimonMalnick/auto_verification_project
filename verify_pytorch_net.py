import os
from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Callable, List
from torch.utils.data import Subset
import torchvision.transforms as trans
import torchvision
from easydict import EasyDict
import numpy as np
import Marabou.maraboupy as maraboupy
import maraboupy.Marabou as pyMarabou
from maraboupy.MarabouNetworkONNX import *
import json
import model_loaders
from tools import torch2marabou_onnx


def parse_args() -> EasyDict:
    convert_str = 'Convert a pytorch network to onnx for Marabou support.'
    verify_str = 'Verify a  pytorch network.'
    ops = ['convert', 'verify', 'both']
    desc = f"Toolbox for verifying pytorch networks using Marabou. the available operations are:\n{ops}\nAn Example of " \
           f"running on my own network from the writeup:\n" \
           f"python verify_pytorch_net.py convert 784 ./tmp.onnx && python verify_pytorch_net.py verify ./tmp.onnx 784\n" \
           f"For more information on a specific operation run:\npython verify_pytorch_net.py <op> -h"
    input_shape_desc = 'The networks input shape (the networks input has to be flatten), for example for mnist it is 784'
    net_loader_desc = 'name of model loading function from the model_loaders.py file'
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter, description=desc)
    sp = parser.add_subparsers()
    convert_op = sp.add_parser('convert', help=convert_str)
    convert_op.set_defaults(cmd='convert')
    convert_op.add_argument('input_shape', type=int, help=input_shape_desc)
    convert_op.add_argument('out_file_path', type=str,
                            help='The full path to the converted network saveing location, e.g. ./mnist_net.onnx')
    convert_op.add_argument('--net_loader', help=net_loader_desc, default='default_loader')

    verify_op = sp.add_parser('verify', help=verify_str)
    verify_op.set_defaults(cmd='verify')
    verify_op.add_argument('net_path', help='path for converted pytorch network (converted using the convert op)')
    verify_op.add_argument('input_shape', type=int, help=input_shape_desc)
    verify_op.add_argument('--timeout', type=int, help='timeout value for Marabou (Default is not timeout)', default=0)
    verify_op.add_argument('--eps', type=float, help='norm bound value', default=0.01)
    verify_op.add_argument('--stats_file_path', help='path to the final statistics file path', default='./stats.json')

    return EasyDict(vars(parser.parse_args()))


def get_net_loader(fn_name: str) -> Callable:
    load_fn = getattr(model_loaders, fn_name, None)
    if load_fn is None:
        raise ValueError('Wrong net_loader function given. Make sure the net_loader is defined in model_loaders.py')
    return load_fn


def get_marabou_net(onnx_path: str, input_names: List[str] = None, output_name: str ='output') -> MarabouNetworkONNX:
    if input_names is None:
        input_names = ['inputs']
    network = pyMarabou.read_onnx(onnx_path, inputNames=input_names, outputName=output_name)
    return network


def run_convert(args: EasyDict):
    load_fn = get_net_loader(args.net_loader)
    assert not os.path.isfile(args.out_file_path) and not os.path.isdir(args.out_file_path)
    torch2marabou_onnx(args, load_fn)
    return f"Saved converted model succesfully to {args.out_file_path}"


def load_random_examples(indices_path):
    with open(indices_path, "r") as in_file:
        indices = [int(line.strip()) for line in in_file.readlines()]
    t = trans.Compose(
        [trans.ToTensor(),
         lambda x: x.numpy().flatten()])
    test_set = torchvision.datasets.MNIST('./mnist/test', train=False, download=True, transform=t)
    test_set_w_indices = Subset(test_set, indices)
    return test_set_w_indices


def run_verify(args: EasyDict):
    marabou_net = get_marabou_net(args.net_path)
    indices_path = "./indices/indices.txt"
    examples = load_random_examples(indices_path)
    outputs = {}
    print("Starting")
    options = pyMarabou.createOptions(numWorkers=8, verbosity=0, timeoutInSeconds=args.timeout)

    for i in range(len(examples)):
        x, y = examples[i]
        vals, stats, maxClass = marabou_net.evaluateLocalRobustness(x, args.eps, y, options=options)
        outputs[i] = {"vals": vals,
                      "maxClass": maxClass,
                      "stats.getTotalTimeInMicro": stats.getTotalTimeInMicro(),
                      "stats.hasTimedOut": stats.hasTimedOut()}
    total, timeouts, not_robust, robust = 0, 0, 0, 0
    all_times = []
    for k in outputs:
        total += 1
        if outputs[k]["stats.hasTimedOut"]:
            timeouts += 1
        elif outputs[k]["vals"]:  # if vals is not empty then an adversarial example was found
            not_robust += 1
        else:  # meaning no timeout and no adv_example, meaning robustness
            robust += 1
        all_times.append(outputs[k]["stats.getTotalTimeInMicro"] * 1e-6)  # convert time to seconds

    all_times = np.array(all_times)
    out_dict = {"timeouts": timeouts,
                "robust": robust,
                "not_robust": not_robust,
                "total": total,
                "median_time": np.median(all_times).item(),
                "min_time": np.min(all_times).item(),
                "max_time": np.max(all_times).item()}
    out_dict.update(outputs)
    with open(args.stats_file_path, "w") as out_j:
        json.dump(out_dict, out_j, indent=4)


def main(args: EasyDict):
    if args.cmd == 'convert':
        run_convert(args)
    elif args.cmd == 'verify':
        run_verify(args)
    elif args.cmd == 'both':
        run_both(args)
    else:
        raise ValueError("Wrong operation given")


if __name__ == '__main__':
    cmd_args = parse_args()
    main(cmd_args)
