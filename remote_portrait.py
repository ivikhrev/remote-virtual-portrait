import logging as log
import sys
from time import perf_counter

from argparse import ArgumentParser

import cv2
import openvino.runtime as ov

from remote_portrait import models
from remote_portrait import meshes

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_argparser():
    parser = ArgumentParser()
    args = parser.add_argument_group('Options')
    args.add_argument('-m_en', '--model_encoder', required=True,
                      help='Required. Path to an .xml file with a trained flame model.')
    args.add_argument('-m_flame', '--model_flame', required=True,
                      help='Required. Path to an .xml file with a trained flame model.')
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, or camera id.')
    args.add_argument('--template', required=True,
                                   help='Required. Path to a head template .obj file.')
    args.add_argument('-d', '--device', default='CPU',
                      help='Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is '
                           'acceptable. The demo will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('-o', '--output', default = 'res.obj',
                         help='Optional. Name of the output .obj file(s) to save.')
    io_args.add_argument('--no_show', action='store_true',
                         help="Optional. Don't show output.")

    return parser


def main():
    args = build_argparser().parse_args()
    start_time = perf_counter()
    log.info(f"Reading image {args.input}")
    img = cv2.imread(args.input) # TODO: crop face using face detector as done in original repo

    core = ov.Core()
    log.info(20*'-' + 'Initialize models' + 20*'-')
    flame_model_params_num = {'shape' : 100, 'tex' : 50, 'exp' : 50, 'pose' : 6, 'cam' : 3, 'light' : 27}
    flame_encoder = models.FlameEncoder(core, args.model_encoder, args.device)
    flame = models.Flame(core, args.model_flame, args.device, flame_model_params_num)

    log.info(20*'-' + 'Encoding input image' + 20*'-')
    parameters = flame_encoder(img)
    log.info(20*'-' + 'Build Flame 3D model' + 20*'-')
    result_dict = flame(parameters)
    meshes.save_obj(args.output, result_dict, args.template)
    end_time = perf_counter()
    log.info(f"Total time: { (end_time - start_time) * 1e3 :.1f} ms")


if __name__ == '__main__':
    sys.exit(main() or 0)
