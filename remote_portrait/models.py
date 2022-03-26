from abc import ABC, abstractmethod
import numpy as np
from skimage.transform import warp, estimate_transform
import cv2

from . import utils

import logging as log


class Model(ABC):
    def __init__(self, core, model_path, device):
        self.core = core
        log.info(f"Reading model {model_path}")
        self.model = core.read_model(model_path)
        utils.log_model_info(self.model)
        log.info(f"Loading model {model_path} to the {device} device")
        self.compiled_model = core.compile_model(self.model, device)
        self.req = self.compiled_model.create_infer_request()
        self.input_names = [input_.get_any_name() for input_ in self.model.inputs]
        self.output_names = [output.get_any_name() for output in self.model.outputs]

    @abstractmethod
    def preprocess(self, input):
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

    def __call__(self, input):
        input_dict = self.preprocess(input)
        output_dict = self.infer(input_dict)
        return self.postrocess(output_dict)


class FlameEncoder(Model):
    def __init__(self, core, model_path, device):
        super().__init__(core, model_path, device)

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
        return output_dict[self.output_names[0]]


class Flame(Model):
    def __init__(self, core, model_path, device, parameters):
        super().__init__(core, model_path, device)
        self.params = parameters

    def preprocess(self, input):
        self.codes = self.decompose_code(input, self.params)
        return { self.input_names[0] : self.codes['shape'],
                 self.input_names[1] : self.codes['exp'],
                 self.input_names[2] : self.codes['pose'] }

    def postrocess(self, output_dict):
        vertices = output_dict[self.output_names[0]]
        landmarks2d = output_dict[self.output_names[0]]
        landmarks3d = output_dict[self.output_names[0]]

        landmarks3d_world = landmarks3d.copy()

        # projection
        landmarks2d = utils.batch_orth_proj(landmarks2d, self.codes['cam'])[:,:,:2];
        landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]

        landmarks3d = utils.batch_orth_proj(landmarks3d, self.codes['cam'])
        landmarks3d[:,:,1:] = -landmarks3d[:,:,1:]

        trans_verts = utils.batch_orth_proj(vertices, self.codes['cam'])
        trans_verts[:,:,1:] = -trans_verts[:,:,1:]

        opdict = {
            'verts': vertices,
            'trans_verts': trans_verts,
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
            'landmarks3d_world': landmarks3d_world,
        }
        return opdict

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