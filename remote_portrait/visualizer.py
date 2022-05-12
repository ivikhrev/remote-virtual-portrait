import logging
from collections import deque
import threading

import cv2
import numpy as np
import pyrender
from regex import R
import trimesh

from .utils import adjust_rect, square_crop_resize

log = logging.getLogger('Global log')


def get_quat(angle, axis):
    half_sin = np.math.sin(0.5 * angle)
    half_cos = np.math.cos(0.5 * angle)
    return (half_sin * axis[0],  # x
            half_sin * axis[1],  # y
            half_sin * axis[2],  # Z
            half_cos)  # W


def get_rotation_from_2vec(vec1, vec2):
    cos_theta = np.dot(np.linalg.norm(vec1), np.linalg.norm(vec2))
    angle = np.math.acos(cos_theta)
    w = np.cross(vec1, vec2)
    w /= np.linalg.norm(w)
    return get_quat(angle, w)


def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.

  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

  return np.array([qx, qy, qz, qw])


def Rx(theta):
  return np.matrix([[1, 0, 0, 0],
                   [0, np.cos(theta),-np.sin(theta), 0],
                   [0, np.sin(theta), np.cos(theta), 0],
                   [0, 0, 0, 1]])

def Ry(theta):
  return np.matrix([[np.cos(theta), 0, np.sin(theta), 0],
                   [0, 1, 0, 0],
                   [-np.sin(theta), 0, np.cos(theta), 0],
                   [0, 0, 0, 1]])

def Rz(theta):
  return np.matrix([[np.cos(theta), -np.sin(theta), 0, 0],
                   [np.sin(theta), np.cos(theta), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0 ,0 , 1]])


class Smoother:
    def __init__(self, smooth_type, seq_len=10):
        self.data_seq = deque(maxlen=seq_len)
        self.avg = None
        if smooth_type == "average":
            self.smooth_func = self.average_smooth
        elif smooth_type == "ema":
            self.smooth_func = self.ema_smooth
        else:
            raise NotImplementedError

    def __call__(self, data, parameters):
        return self.smooth_func(data, parameters)

    def avg_smooth(self, data, parameters=None):
        self.rect_seq.append(data)
        return np.array(sum(self.data_seq) / len(self.data_seq), dtype=data.dtype)

    def ema_smooth(self, data, alpha=0.5):
        if self.avg is not None:
            self.avg = alpha * data + (1 - alpha) * self.avg
        else:
            self.avg = data
        return self.avg.astype(data.dtype)


class AnimationThread(threading.Thread):
    def __init__(self, render_lock, scene, head, head_mesh, parameters, face_detector, flame_encoder, flame):
        threading.Thread.__init__(self)
        self.render_lock = render_lock
        self.scene = scene
        self.head = head
        self.head_mesh = head_mesh
        self.initial_rotation = parameters["pose"][0][:3]
        self.parameters = parameters

        self.face_detector = face_detector
        self.flame_encoder = flame_encoder
        self.flame = flame
        self.cap = cv2.VideoCapture(0)

        self.rectangle_smoother = Smoother("ema")
        self.rotation_smoother = Smoother("ema")

    def run(self):
        while True:
            _, img = self.cap.read()
            h, w, _ = img.shape
            detections = self.face_detector(img)

            if len(detections) == 1:
                face = detections[0]
                bottom_left, top_right = adjust_rect(face.xmin, face.ymin,
                    face.xmax, face.ymax, 0.15, 0.2)
                bottom_left, top_right = self.rectangle_smoother(np.array([bottom_left, top_right]), 0.3)
                cropped_face = square_crop_resize(img, bottom_left, top_right, 224)
                cv2.rectangle(img, bottom_left, top_right, (0, 255, 0))
                parameters = self.flame_encoder(cropped_face)
                # transfer only mouth movement and expression
                self.parameters['pose'][0][3:] = parameters['pose'][0][3:]
                self.parameters['exp'] = parameters['exp']
                result_dict = self.flame(self.parameters)

                # modify meshes
                self.head_mesh.vertices = result_dict['verts'][0]
                head_mesh = pyrender.Mesh.from_trimesh(self.head_mesh)
                # determine transform parameters
                pose = parameters['pose'][0][:3] - self.initial_rotation
                pose = self.rotation_smoother(pose, 0.6)
                # transform_matrix = np.asarray(Rx(pose[0]) * Ry(pose[1]) * Rz(pose[2]))
                # offset = np.array([0, 0.09, 0.018])
                # transform_matrix[:3,3] = self.head.mesh.centroid + offset
                rot = get_quaternion_from_euler(*pose)

                self.render_lock.acquire()
                self.head.mesh = head_mesh
                self.head.rotation = rot
                #self.hairs.matrix = transform_matrix.tolist()
                self.render_lock.release()
            cv2.imshow("img", img)
            cv2.waitKey(1)


# pylint: disable=W0223
class Visualizer:
    def __init__ (self, mesh_obj_filename, window_size, orig_params, face_detector, flame_encoder, flame):
        self.size = window_size
        self.parameters = orig_params
        self.face_detector = face_detector
        self.flame_encoder = flame_encoder
        self.flame = flame

        # hairs_obj_filename = "C:/Users/ivikhrev/Documents/Study/diploma/remote-portrait/resources/blond_hairs.obj"
        # hairs_mesh = trimesh.load(hairs_obj_filename)
        # hairs_mesh = pyrender.Mesh.from_trimesh(hairs_mesh)
        # self.hairs = pyrender.Node(mesh=hairs_mesh, matrix=np.eye(4))

        self.head_mesh = trimesh.load(mesh_obj_filename, process=False, maintain_order=True)
        head_mesh = pyrender.Mesh.from_trimesh(self.head_mesh)
        self.head = pyrender.Node(mesh=head_mesh, matrix=np.eye(4))
        #offset = np.array([0, 0.09, 0.018])
        #self.hairs.translation = self.head.mesh.centroid + offset #[np.average(vertices[:,0]), np.average(vertices[:,1]) + 0.05, np.average(vertices[:,2]) - 0.04]#[np.min(vertices[:,0]), np.max(vertices[:,1]), -0.04]

        self.scene = pyrender.Scene(bg_color=(0,0,0,0), ambient_light=(255, 255, 255))
        #self.scene.add_node(self.hairs)
        self.scene.add_node(self.head)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
        camera_pos = np.eye(4)
        camera_pos[2][3] = .5
        self.scene.add(camera, pose=camera_pos)

    def run(self):
        v = pyrender.Viewer(self.scene, viewport_size=self.size, use_raymond_lighting=True, run_in_thread=True)
        animation_thread = AnimationThread(v._render_lock, self.scene, self.head, self.head_mesh, self.parameters,
            self.face_detector, self.flame_encoder, self.flame)
        animation_thread.start()
