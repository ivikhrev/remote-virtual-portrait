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
    args.add_argument('-m_fd', '--model_face_detector', required=True,
                      help='Required. Path to an .xml file with a trained face detection model.')
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


def adjust_rect(xmin, ymin, xmax, ymax, coeffx=0.1, coeffy=0.1, offset_x = 0, offset_y = 0):
    h , w =  ymax - ymin, xmax - xmin
    dx = int(w * coeffx / 2)
    dy = int(h * coeffy / 2)
    return (xmin - dx + offset_x, ymin - dy + offset_y), (xmax +  dx + offset_x, ymax + dy + offset_y)

def main():
    args = build_argparser().parse_args()
    start_time = perf_counter()
    log.info(f"Reading image {args.input}")
    img = cv2.imread(args.input) # TODO: crop face using face detector as done in original repo

    core = ov.Core()
    log.info(20*'-' + 'Initialize models' + 20*'-')
    face_model = models.UltraLightFace(core, args.model_face_detector, args.device)
    detections = face_model(img)
    print(len(detections))
    for d in detections:
        print(d.bottom_left_point(), d.top_right_point())
        print(d.score)
        bottom_left, top_right = adjust_rect(d.xmin, d.ymin, d.xmax, d.ymax, 0.4, 0.35,  0, -20)
        cv2.rectangle(img,  bottom_left, top_right, (255,0,0), 2)
        h, w = top_right[1] - bottom_left[1], top_right[0] - bottom_left[0]
        print(h, w)
        if h > w: # h > w
            offset = h - w
            print(bottom_left[0] - offset // 2, top_right[0] + offset // 2, bottom_left[1], top_right[1])
            cv2.imshow("test2", img[bottom_left[1] : top_right[1], bottom_left[0] - offset // 2:top_right[0] + offset // 2]) # h x w
            crop = img[bottom_left[1] : top_right[1], bottom_left[0] - offset // 2:top_right[0] + offset // 2]
        else:
            offset = w - h
            cv2.imshow("test2", img[bottom_left[1] - offset // 2:top_right[1] + offset // 2, bottom_left[0]:top_right[0]])
        cv2.waitKey(0)
    cv2.imshow("test", img)
    cv2.waitKey(0)
    img = cv2.resize(crop, dsize=(224, 224))
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
