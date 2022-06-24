import logging
from time import perf_counter
from pathlib import Path

import cv2
import openvino.runtime as ov
import numpy as np
from pytorch3d import io

# pylint: disable=C0413
from .models import SFD, UltraLightFace, FlameEncoder, DetailEncoder, DetailDecoder, Flame, FlameAlbedo
from .meshes import save_obj
from .utils import adjust_rect, square_crop_resize
from .images_capture import open_images_capture
from .texture_renderer import Texture


log = logging.getLogger('Global log')


def reconstruct(config):
    start_time = perf_counter()

    log.info(20 * '-' + 'Initialize models' + 20 * '-')
    device = config["device"]
    core = ov.Core()
    face_detector = SFD(core, config["face_detector"], device)
    fast_face_detector = UltraLightFace(core, config["fast_face_detector"], config["device_fd"])
    flame_encoder = FlameEncoder(core, config["flame_encoder"], config["device_flame_en"])
    detail_encoder = DetailEncoder(core, config["details_encoder"], device)
    detail_decoder = DetailDecoder(core,config["details_decoder"], device)
    flame = Flame(core, config["flame"], config["device_flame"])
    flame_albedo = FlameAlbedo(core, config["flame_texture"], device)

    input_path = config['test_input']
    input_name = Path(config['test_input']).stem

    cap = open_images_capture(input_path, True)

    img = cap.read()
    detections = face_detector(img)
    if len(detections) == 1:
        face = detections[0]
        bottom_left, top_right = adjust_rect(face.xmin, face.ymin,
            face.xmax, face.ymax, 0.15, 0.2)
        cropped_face = square_crop_resize(img, bottom_left, top_right, 224)
        cv2.rectangle(img, bottom_left, top_right, (255, 0, 0), 2)
        # cv2.imshow("orig", img)
        # cv2.waitKey(0)
    elif len(detections) > 1:
        raise Exception("On image must be only one face")
    else:
        raise Exception("Can't detect any face")

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
    albedo = flame_albedo(parameters['tex'])
    # cv2.imshow("albedo", cv2.cvtColor(np.float32(albedo.transpose(0, 2, 3, 1)[0]), cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    tex = uvfaces = uvcoords = None

    tex_start_time = perf_counter()
    if config["use_tex"]:
        texture_generator = Texture(io.load_obj(config["head_template"]),
            np.load(config["fixed_uv_displacement"]),
            cv2.imread(config["uv_face_eye_mask"], cv2.IMREAD_GRAYSCALE))

        tex, uvcoords, uvfaces = texture_generator(cropped_face, albedo, uv_z,
            result_dict['verts'], result_dict['trans_verts'], parameters['light'])
        uvfaces = uvfaces[0]
        uvcoords = uvcoords[0]
    log.info(f"Texture generation time: { (perf_counter() -  tex_start_time) * 1e3 :.1f} ms")

    resulted_obj_path = config["output_path"] + '/' + input_name + '.obj'
    save_obj(resulted_obj_path, result_dict,
        config["head_template"], tex, uvcoords, uvfaces)


    end_time = perf_counter()
    log.info(f"Total time: { (end_time - start_time) * 1e3 :.1f} ms")
    return resulted_obj_path, parameters, fast_face_detector, flame_encoder, flame