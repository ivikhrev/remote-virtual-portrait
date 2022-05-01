import logging
import numpy as np
import cv2


log = logging.getLogger('Global log')


class Detection:
    def __init__(self, xmin, ymin, xmax, ymax, score, class_id):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.score = score
        self.id = class_id

    def bottom_left_point(self):
        return self.xmin, self.ymin

    def top_right_point(self):
        return self.xmax, self.ymax

    def get_coords(self):
        return self.xmin, self.ymin, self.xmax, self.ymax


def resize_image(image, size, keep_aspect_ratio=False, interpolation=cv2.INTER_LINEAR):
    if not keep_aspect_ratio:
        resized_frame = cv2.resize(image, size, interpolation=interpolation)
    else:
        h, w = image.shape[:2]
        scale = min(size[1] / h, size[0] / w)
        resized_frame = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)
    return resized_frame


def log_model_info(model):
    log.info(f"Model name: {model.get_name()}")
    log.info("Inputs:")
    for input_ in model.inputs:
        log.info(f"\t{input_.get_any_name()} : shape {input_.partial_shape}")
    log.info("Outputs:")
    for output_ in model.outputs:
        log.info(f"\t{output_.get_any_name()} : shape {output_.partial_shape}")


def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.copy().reshape((-1, 1, 3))
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = np.concatenate([X_trans, X[:,:,2:]], axis=2)
    #shape = X_trans.shape
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn


def nms(x1, y1, x2, y2, scores, thresh, keep_top_k=None):
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    if keep_top_k:
        order = order[:keep_top_k]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        union = (areas[i] + areas[order[1:]] - intersection)
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)
        order = order[np.where(overlap <= thresh)[0] + 1]

    return keep
