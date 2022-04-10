import logging
from abc import ABC, abstractmethod
import numpy as np
from skimage.transform import warp, estimate_transform
import cv2

from .utils import Detection, batch_orth_proj, log_model_info, nms, resize_image


log = logging.getLogger('Global log')


class Model(ABC):
    def __init__(self, core, model_path, device):
        self.core = core
        log.info(f"Reading model {model_path}")
        self.model = core.read_model(model_path)
        log_model_info(self.model)
        log.info(f"Loading model {model_path} to the {device} device")
        self.compiled_model = core.compile_model(self.model, device)
        self.req = self.compiled_model.create_infer_request()
        self.input_names = [input_.get_any_name() for input_ in self.model.inputs]
        self.output_names = [output_.get_any_name() for output_ in self.model.outputs]

    @abstractmethod
    def preprocess(self, model_input):
        raise NotImplementedError("Base abstract class method must be overriden!")

    @abstractmethod
    def postrocess(self, output_dict):
        raise NotImplementedError("Base abstract class method must be overriden!")

    def infer(self, input_dict):
        output = self.req.infer(input_dict)
        output_dict = {}
        for out_tensor in self.compiled_model.outputs:
            output_dict[out_tensor.get_any_name()] = output[out_tensor]
        return output_dict

    def __call__(self, model_input):
        input_dict = self.preprocess(model_input)
        output_dict = self.infer(input_dict)
        return self.postrocess(output_dict)


class UltraLightFace(Model):
    def __init__(self, core, model_path, device, conf_t=0.5, iou_t=0.5):
        super().__init__(core, model_path, device)
        self.image_blob_name = self.input_names[0]
        self.n, self.c, self.h, self.w = self.model.input().get_shape()
        self.scores_tensor_name = self.output_names[0]
        self.bboxes_tensor_name = self.output_names[1]
        self.confidence_threshold = conf_t
        self.iou_threshold = iou_t

    def preprocess(self, image):
        self.original_shape = image.shape
        resized_image = resize_image(image, (self.w, self.h))
        self.resized_shape = resized_image.shape
        resized_image = resized_image.transpose((2, 0, 1))  # HWC->CHW
        resized_image = resized_image.reshape((1, self.c, self.h, self.w))
        return  {self.image_blob_name : resized_image}

    def postrocess(self, output_dict):
        boxes = output_dict[self.bboxes_tensor_name][0]
        scores = output_dict[self.scores_tensor_name][0]

        score = np.transpose(scores)[1]

        mask = score > self.confidence_threshold
        filtered_boxes, filtered_score = boxes[mask, :], score[mask]

        x_mins, y_mins, x_maxs, y_maxs = filtered_boxes.T

        keep = nms(x_mins, y_mins, x_maxs, y_maxs, filtered_score, self.iou_threshold)

        filtered_score = filtered_score[keep]
        x_mins, y_mins, x_maxs, y_maxs = x_mins[keep], y_mins[keep], x_maxs[keep], y_maxs[keep]
        detections = [Detection(*det, 0) for det in zip(x_mins, y_mins, x_maxs, y_maxs, filtered_score)]
        detections = UltraLightFace.resize_detections(detections, self.original_shape[1::-1])
        return UltraLightFace.clip_detections(detections, self.original_shape)

    @staticmethod
    def resize_detections(detections, original_image_size):
        for detection in detections:
            detection.xmin *= original_image_size[0]
            detection.xmax *= original_image_size[0]
            detection.ymin *= original_image_size[1]
            detection.ymax *= original_image_size[1]
        return detections

    @staticmethod
    def clip_detections(detections, size):
        for detection in detections:
            detection.xmin = max(int(detection.xmin), 0)
            detection.ymin = max(int(detection.ymin), 0)
            detection.xmax = min(int(detection.xmax), size[1])
            detection.ymax = min(int(detection.ymax), size[0])
        return detections


class FlameEncoder(Model):
    def __init__(self, core, model_path, device):
        super().__init__(core, model_path, device)
        self.params_num = {'shape' : 100,
                            'tex' : 50,
                            'exp' : 50,
                            'pose' : 6,
                            'cam' : 3,
                            'light' : 27}

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.
        h, w, _ = image.shape
        resolution_inp = 224
        src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
        dst_pts = np.array([[0,0], [0,resolution_inp - 1], [resolution_inp - 1, 0]])
        tranformation = estimate_transform('similarity', src_pts, dst_pts)
        transformed_image = warp(image, tranformation.inverse, output_shape=(h, w))
        transformed_image = transformed_image.transpose(2,0,1)
        input_tensor = np.expand_dims(transformed_image , 0)
        return {self.input_names[0] : input_tensor}

    def postrocess(self, output_dict):
        return FlameEncoder.decompose_code(output_dict[self.output_names[0]], self.params_num)

    @staticmethod
    def decompose_code(code, params_num):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in params_num:
            end = start + params_num[key]
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict


class Flame(Model):
    def preprocess(self, model_input):
        self.codes = model_input
        return { self.input_names[0] : self.codes['shape'],
                 self.input_names[1] : self.codes['exp'],
                 self.input_names[2] : self.codes['pose'] }

    def postrocess(self, output_dict):
        vertices = output_dict[self.output_names[0]]
        landmarks2d = output_dict[self.output_names[0]]
        landmarks3d = output_dict[self.output_names[0]]

        landmarks3d_world = landmarks3d.copy()

        # projection
        landmarks2d = batch_orth_proj(landmarks2d, self.codes['cam'])[:,:,:2]
        landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]

        landmarks3d = batch_orth_proj(landmarks3d, self.codes['cam'])
        landmarks3d[:,:,1:] = -landmarks3d[:,:,1:]

        trans_verts = batch_orth_proj(vertices, self.codes['cam'])
        trans_verts[:,:,1:] = -trans_verts[:,:,1:]

        opdict = {
            'verts': vertices,
            'trans_verts': trans_verts,
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
            'landmarks3d_world': landmarks3d_world,
        }
        return opdict


class FlameTexture(Model):
    def __init__(self, core, model_path, device, tex_params):
        super().__init__(core, model_path, device)
        self.tex_params = tex_params

    def preprocess(self, model_input):
        self.codes = self.decompose_code(model_input, self.params)
        return { self.input_names[0] : self.codes}

    def postrocess(self, output_dict):
        """"""
