from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from easydict import EasyDict
from model_loaders import *


def parse_args() -> EasyDict:
    convert_str = 'Convert a pytorch network to onnx for Marabou support. requires the arguments:\ninput_shape\n' \
                  'out_file_path'
    verify_str = 'Verify a  pytorch network, requires the arguments: converted_path\ninput_shape'
    ops = ['convert', 'verify']
    desc = f"Toolbox for verifying pytorch networks using Marabou. the available operations are: {ops}"
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description=desc)
    sp = parser.add_subparsers()
    convert_op = sp.add_parser('convert', help=convert_str)
    convert_op.set_defaults(cmd='convert')
    convert_op.add_argument('input_shape', type=int, help='The networks input shape (the networks input has to be flatten), for example for mnist it is 784')
    convert_op.add_argument('out_file_path', type=str, help='The full path to the converted network saveing location, e.g. ./mnist_net.onnx')
    convert_op.add_argument('--net_loader', help='name of model loading function for the model_loaders.py file',
                            default='default_loader')
    verify_op = sp.add_parser('verify', help=verify_str)
    verify_op.set_defaults(cmd='verify')
    verify_op.add_argument('net_path', help='path for converted pytorch network (converted using the convert op)')
    verify_op.add_argument('input_shape', type=int, help='The networks input shape (the networks input has to be flatten), for example for mnist it is 784')

    return EasyDict(vars(parser.parse_args()))

