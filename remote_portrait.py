import logging
from time import perf_counter
import sys

from argparse import ArgumentParser

import cv2
import openvino.runtime as ov
import numpy as np
from pyglet.canvas import get_display


log = logging.getLogger('Global log')
log_handler = logging.StreamHandler()
log.addHandler(log_handler)
log.setLevel(logging.DEBUG)
log_handler.setLevel(logging.DEBUG)
log_handler.setFormatter(logging.Formatter('[ %(levelname)s ] %(message)s'))


try:
    log.info("Trying to get display...")
    display = get_display()
    log.info("Successfully")
except Exception:
    log.info("Can't get physical display. Create virtual one.")
    from pyvirtualdisplay import Display
    virtual_display = Display(visible=0, size=(1920, 1080))
    virtual_display.start()

from remote_portrait import models, meshes
from remote_portrait.visualizer import Visualizer


def build_argparser():
    parser = ArgumentParser()
    args = parser.add_argument_group('Options')
    args.add_argument('-m_fd', '--model_face_detector', required=True,
                      help='Required. Path to an .xml file with a trained face detection model.')
    args.add_argument('-m_en', '--model_encoder', required=True,
                      help='Required. Path to an .xml file with a trained flame model.')
    args.add_argument('-m_flame', '--model_flame', required=True,
                      help='Required. Path to an .xml file with a trained flame model.')
    args.add_argument('-m_tex', '--model_texture', required=True,
                      help='Required. Path to an .xml file with a trained texture model.')
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


def adjust_rect(xmin, ymin, xmax, ymax, coeffx=0.1, coeffy=0.1, offset_x = 0, offset_y = 0):
    h , w =  ymax - ymin, xmax - xmin
    dx = int(w * coeffx / 2)
    dy = int(h * coeffy / 2)
    return (xmin - dx + offset_x, ymin - dy + offset_y), (xmax +  dx + offset_x, ymax + dy + offset_y)


def square_crop_resize(img, bottom_left_point, top_right_point, target_size):
    orig_h, orig_w, _ = img.shape
    bottom_left_point = (np.clip(bottom_left_point[0], 0, orig_w), np.clip(bottom_left_point[1], 0, orig_h))
    top_right_point = (np.clip(top_right_point[0], 0, orig_w), np.clip(top_right_point[1], 0, orig_h))

    h, w = top_right_point[1] - bottom_left_point[1], top_right_point[0] - bottom_left_point[0]

    if h > w:
        offset = h - w
        crop = img[bottom_left_point[1] : top_right_point[1], bottom_left_point[0]
            - offset // 2:top_right_point[0] + offset // 2]
    else:
        offset = w - h
        crop = img[bottom_left_point[1] - offset // 2:top_right_point[1]
            + offset // 2, bottom_left_point[0]:top_right_point[0]]
    return cv2.resize(crop, dsize=(target_size, target_size))


def main():
    args = build_argparser().parse_args()
    start_time = perf_counter()
    log.info(f"Reading image {args.input}")
    img = cv2.imread(args.input)
    if img is None:
        raise ValueError(f"Can't read image {args.input}")
    core = ov.Core()
    log.info(20*'-' + 'Crop face from image' + 20*'-')
    face_model = models.UltraLightFace(core, args.model_face_detector, args.device)
    detections = face_model(img)

    if len(detections) == 0:
        raise RuntimeError("No face detected!")
    elif len(detections) > 1:
        raise RuntimeError("More than 1 face detected! Please provide image with 1 face only!")
    face = detections[0]

    bottom_left, top_right = adjust_rect(face.xmin, face.ymin,
        face.xmax, face.ymax, 0.4, 0.35, 0, -20)

    cropped_face = square_crop_resize(img, bottom_left, top_right, 224)
    if not args.no_show:
        cv2.imshow("cropped face", cropped_face)

    log.info(20*'-' + 'Initialize models' + 20*'-')
    flame_encoder = models.FlameEncoder(core, args.model_encoder, args.device)
    flame = models.Flame(core, args.model_flame, args.device)

    log.info(20*'-' + 'Encoding input image' + 20*'-')
    parameters = flame_encoder(cropped_face)
    log.info(20*'-' + 'Build Flame 3D model' + 20*'-')
    result_dict = flame(parameters)
    meshes.save_obj(args.output, result_dict, args.template)
    visualizer = Visualizer(800, 600, args.output, not args.no_show)
    visualizer.run()
    end_time = perf_counter()
    log.info(f"Total time: { (end_time - start_time) * 1e3 :.1f} ms")


if __name__ == '__main__':
    sys.exit(main() or 0)
