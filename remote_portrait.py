import logging
from time import perf_counter
import sys

from argparse import ArgumentParser

import cv2
from cv2 import rectangle
import openvino.runtime as ov
import numpy as np
from pytorch3d import io

log = logging.getLogger('Global log')
log_handler = logging.StreamHandler()
log.addHandler(log_handler)
log.setLevel(logging.DEBUG)
log_handler.setLevel(logging.DEBUG)
log_handler.setFormatter(logging.Formatter('[ %(levelname)s ] %(message)s'))


# pylint: disable=C0413
from remote_portrait import models, meshes
from remote_portrait.config import Config
from remote_portrait.images_capture import open_images_capture
from remote_portrait.texture import Texture
from remote_portrait.visualizer import Visualizer


def build_argparser():
    parser = ArgumentParser()
    args = parser.add_argument_group('Options')
    args.add_argument('-c', '--config', required=True,
                      help='Required. Path to a config file.')
    return parser


def adjust_rect(xmin, ymin, xmax, ymax, coeffx=0.1, coeffy=0.1):
    h , w =  ymax - ymin, xmax - xmin
    dx = int(w * coeffx)
    dy = int(h * coeffy)
    return (xmin - dx , ymin), (xmax + dx , ymax + dy)


def square_crop_resize(img, bottom_left_point, top_right_point, target_size):
    orig_h, orig_w, _ = img.shape
    bottom_left_point = (np.clip(bottom_left_point[0], 0, orig_w),
        np.clip(bottom_left_point[1], 0, orig_h))
    top_right_point = (np.clip(top_right_point[0], 0, orig_w),
        np.clip(top_right_point[1], 0, orig_h))

    h, w = top_right_point[1] - bottom_left_point[1], top_right_point[0] - bottom_left_point[0]

    if h > w:
        offset = h - w
        crop = img[bottom_left_point[1]:top_right_point[1], np.clip(bottom_left_point[0]
            - offset // 2, 0, orig_w):np.clip(top_right_point[0] + offset // 2, 0, orig_w)]
    else:
        offset = w - h
        crop = img[np.clip(bottom_left_point[1] - offset // 2, 0, orig_h):np.clip(top_right_point[1]
            + offset // 2, 0, orig_h), bottom_left_point[0]:top_right_point[0]]

    return cv2.resize(crop, dsize=(target_size, target_size))


def main():
    args = build_argparser().parse_args()
    config = Config(file_path=args.config)
    start_time = perf_counter()

    pose = {
        "yaw" : 0,
        "pitch": 0,
        "roll": 0
    }

    log.info(20 * '-' + 'Initialize models' + 20 * '-')
    device = config.properties["device"]
    core = ov.Core()
    face_detector = models.SFD(core, config.properties["face_detector"], device)
    head_pose_estimator = models.HeadPoseEstimation(core, config.properties["head_pose"], device)
    flame_encoder = models.FlameEncoder(core, config.properties["flame_encoder"], device)
    detail_encoder = models.DetailEncoder(core, config.properties["details_encoder"], device)
    detail_decoder = models.DetailDecoder(core,config.properties["details_decoder"], device)
    flame = models.Flame(core, config.properties["flame"], device)
    flame_texture = models.FlameTexture(core, config.properties["flame_texture"], device)

    input_name = config.properties['test_input']

    cap = open_images_capture(input_name, True)
    delay = int(cap.get_type() in {'VIDEO', 'CAMERA'})

    while True:
        img = cap.read()
        h, w, _ = img.shape
        detections = face_detector(img)
        if len(detections) == 1:
            face = detections[0]
            bottom_left, top_right = adjust_rect(face.xmin, face.ymin,
                face.xmax, face.ymax, 0.15, 0.2)
            cropped_face = square_crop_resize(img, bottom_left, top_right, 224)
            cv2.rectangle(img, bottom_left, top_right, (255, 0, 255))
        elif len(detections) == 0:
            cv2.putText(img, "No face detected!", (int(w/3), int(4/5 *h)),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
        else:
            cv2.putText(img, "More than 1 face detected! Please provide image with 1 face only!",
                (int(w/3), int(4/5 *h)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))

        cv2.imshow("face", img)
        key = cv2.waitKey(delay)
        if key in {ord('q'), ord('Q'), 27}:
            break

    cv2.destroyAllWindows()


    if config.properties["pose_image"]:
        pose_img = cv2.imread(config.properties["pose_image"])
        pose = head_pose_estimator(pose_img)

    print("Head pose:", pose)

    # bottom_left, top_right = adjust_rect(face.xmin, face.ymin,
    #    face.xmax, face.ymax, 0.15, 0.2)

    # cropped_face = square_crop_resize(img, bottom_left, top_right, 224)

    # if not config.properties["no_show"]:
    #     cv2.imshow("cropped face", cropped_face)

    log.info(20*'-' + 'Encode input image' + 20*'-')
    parameters = flame_encoder(cropped_face)

    log.info(20*'-' + 'Encode details' + 20*'-')
    details = detail_encoder(cropped_face)
    parameters['details'] = details

    log.info(20*'-' + 'Decode details model output' + 20*'-')
    uv_z = detail_decoder(np.concatenate((parameters['pose'][:,3:],
        parameters['exp'], parameters['details']), axis=1))

    log.info(20*'-' + 'Build Flame 3D model' + 20*'-')
    result_dict = flame(parameters)

    log.info(20*'-' + 'Create Flame texture' + 20*'-')
    albedo = flame_texture(parameters['tex'])

    tex = uvfaces = uvcoords = None
    if config.properties["use_tex"]:
        texture_generator = Texture(io.load_obj(config.properties["head_template"]),
            np.load(config.properties["fixed_uv_displacement"]),
            cv2.imread(config.properties["uv_face_eye_mask"], cv2.IMREAD_GRAYSCALE))

        tex, uvcoords, uvfaces = texture_generator(cropped_face, albedo, uv_z,
            result_dict['verts'], result_dict['trans_verts'], parameters['light'])
        uvfaces = uvfaces[0]
        uvcoords = uvcoords[0]

    meshes.save_obj(config.properties["output_name"], result_dict,
        config.properties["head_template"], tex, uvcoords, uvfaces)

    visualizer = Visualizer(config.properties["output_name"], config.properties["visualizer_size"], face_detector, head_pose_estimator)
    visualizer.run()

    end_time = perf_counter()
    log.info(f"Total time: { (end_time - start_time) * 1e3 :.1f} ms")


if __name__ == '__main__':
    sys.exit(main() or 0)
