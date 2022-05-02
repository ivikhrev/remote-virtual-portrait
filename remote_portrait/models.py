import logging
from abc import ABC, abstractmethod
import numpy as np
from skimage.transform import warp, estimate_transform
import cv2

from .utils import Detection, batch_orth_proj, log_model_info, nms, resize_image


log = logging.getLogger('Global log')


class Model(ABC):
    def __init__(self, core, model_path, device, log_info=True, compile_model=True):
        self.core = core
        log.info(f"Reading model {model_path}")
        self.model = core.read_model(model_path)
        if log_info:
            log_model_info(self.model)
        if compile_model:
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


class SFD(Model):
    def __init__(self, core, model_path, device, conf_t=0.5, iou_t=0.5):
        super().__init__(core, model_path, device, log_info=False, compile_model=False)
        self.model.reshape((1,3,-1,-1))
        log_model_info(self.model)
        log.info(f"Loading model {model_path} to the {device} device")
        self.compiled_model = core.compile_model(self.model, device)
        self.req = self.compiled_model.create_infer_request()
        self.image_blob_name = self.input_names[0]
        self.n, self.c, self.h, self.w = self.model.input().get_partial_shape()
        print(self.h, self.w)
        self.scores_tensor_name = self.output_names[0]
        self.bboxes_tensor_name = self.output_names[1]
        self.confidence_threshold = conf_t
        self.iou_threshold = iou_t

    def preprocess(self, image):
        self.original_shape = image.shape
        input_image = image.copy()
        input_image = input_image.transpose((2, 0, 1))  # HWC->CHW
        input_image = np.expand_dims(input_image, 0)
        return  {self.image_blob_name : input_image}

    def postrocess(self, output_dict):
        raw_results = list(output_dict.values())

        for i in range(len(raw_results) // 2):
            raw_results[i * 2] = SFD.softmax(raw_results[i * 2], axis=1)

        boxes = SFD.get_predictions(raw_results)
        filtered_boxes = boxes[boxes[:,4] > self.confidence_threshold]
        keep = nms(filtered_boxes[:,0], filtered_boxes[:,1], filtered_boxes[:,2],
            filtered_boxes[:,3], filtered_boxes[:,4], self.iou_threshold)
        filtered_boxes = filtered_boxes[keep]
        detections = [Detection(*det, 0) for det in filtered_boxes]

        return UltraLightFace.clip_detections(detections, self.original_shape)

    @staticmethod
    def get_predictions(olist):
        variances = [0.1, 0.2]
        bboxlist = []
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            stride = 2**(i + 2)  # 4,8,16,32,64,128
            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
            for _, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[0, 1, hindex, windex]
                loc = oreg[0, :, hindex, windex].copy().reshape(1, 4)
                priors = np.array([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                box = SFD.decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0]
                bboxlist.append([x1, y1, x2, y2, score])

        return np.array(bboxlist)

    @staticmethod
    def decode(loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """

        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    @staticmethod
    def softmax(logits, axis=None):
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=axis)


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
        src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])
        dst_pts = np.array([[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
        tranformation = estimate_transform('similarity', src_pts, dst_pts)
        transformed_image = warp(image, tranformation.inverse, output_shape=(h, w))
        transformed_image = transformed_image.transpose(2, 0, 1)
        input_tensor = np.expand_dims(transformed_image, 0)
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


class DetailEncoder(FlameEncoder):
    def postrocess(self, output_dict):
        return output_dict[self.output_names[0]]


class DetailDecoder(Model):
    def preprocess(self, model_input):
        return { self.input_names[0] : model_input }

    def postrocess(self, output_dict):
        return output_dict[self.output_names[0]]


class Flame(Model):
    def preprocess(self, model_input):
        self.codes = model_input
        return { self.input_names[0] : self.codes['shape'],
                 self.input_names[1] : self.codes['exp'],
                 self.input_names[2] : self.codes['pose'] }

    def postrocess(self, output_dict):
        vertices = output_dict[self.output_names[0]]
        landmarks2d = output_dict[self.output_names[1]]  # maybe convert? models with names specifying
        landmarks3d = output_dict[self.output_names[2]]

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
    def preprocess(self, model_input):
        return { self.input_names[0] : model_input }

    def postrocess(self, output_dict):
        return output_dict[self.output_names[0]]
