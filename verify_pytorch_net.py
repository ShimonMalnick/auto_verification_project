from argparse import ArgumentParser, RawTextHelpFormatter
from easydict import EasyDict
from typing import Callable
import model_loaders
from tools import torch2marabou_onnx
import os


def parse_args() -> EasyDict:
    convert_str = 'Convert a pytorch network to onnx for Marabou support.'
    verify_str = 'Verify a  pytorch network.'
    ops = ['convert', 'verify', 'both']
    desc = f"Toolbox for verifying pytorch networks using Marabou. the available operations are:\n{ops}\nAn Example of " \
           f"running on my own network from the writeup:\n" \
           f"python verify_pytorch_net.py both --input_shape 784\n" \
           f"For more information on a specific operation run:\npython verify_pytorch_net.py <op> -h"
    input_shape_desc = 'The networks input shape (the networks input has to be flatten), for example for mnist it is 784'
    net_loader_desc = 'name of model loading function from the model_loaders.py file'
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter, description=desc)
    sp = parser.add_subparsers()
    convert_op = sp.add_parser('convert', help=convert_str)
    convert_op.set_defaults(cmd='convert')
    convert_op.add_argument('input_shape', type=int, help=input_shape_desc)
    convert_op.add_argument('out_file_path', type=str, help='The full path to the converted network saveing location, e.g. ./mnist_net.onnx')
    convert_op.add_argument('--net_loader', help=net_loader_desc, default='default_loader')

    verify_op = sp.add_parser('verify', help=verify_str)
    verify_op.set_defaults(cmd='verify')
    verify_op.add_argument('net_path', help='path for converted pytorch network (converted using the convert op)')
    verify_op.add_argument('input_shape', type=int, help=input_shape_desc)

    both_op = sp.add_parser('both', help='Run both operations sequentially')
    both_op.set_defaults(cmd='both')
    both_op.add_argument('input_shape', type=int, help=input_shape_desc)
    both_op.add_argument('--net_loader', help=net_loader_desc, default='default_loader')

    return EasyDict(vars(parser.parse_args()))


def get_net_loader(fn_name: str) -> Callable:
    load_fn = getattr(model_loaders, fn_name, None)
    if load_fn is None:
        raise ValueError('Wrong net_loader function given. Make sure the net_loader is defined in model_loaders.py')
    return load_fn


def run_convert(args):
    load_fn = get_net_loader(args.net_loader)
    assert not os.path.isfile(args.out_file_path) and not os.path.isdir(args.out_file_path)
    torch2marabou_onnx(args, load_fn)
    return f"Saved converted model succesfully to {args.out_file_path}"


def run_verify(args):
    pass


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


