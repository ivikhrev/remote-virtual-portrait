import logging
from math import sin, cos, radians

import cv2
import numpy as np
import pyrender
import trimesh


log = logging.getLogger('Global log')


def euler_to_rotation_matrix(yaw, pitch, roll, angles=False):
    if angles:
        yaw = radians(yaw)
        pitch = radians(pitch)
        roll = radians(roll)

    res = np.eye(4)
    res[0][0] = cos(yaw) * cos(pitch)
    res[0][1] = cos(yaw) * sin(pitch) * sin(roll) - sin(yaw) * cos(roll)
    res[0][2] = cos(yaw) * sin(pitch) * cos(roll) + sin(yaw) * sin(roll)
    res[1][0] = sin(yaw) * cos(pitch)
    res[1][1] = sin(yaw) * sin(pitch) * sin(roll) + cos(yaw) * cos(roll)
    res[1][2] = sin(yaw) * sin(pitch) * cos(roll) - cos(yaw) * sin(roll)
    res[2][0] = -sin(pitch)
    res[2][1] = cos(pitch) * sin(roll)
    res[2][2] = cos(pitch) * cos(roll)

    return res


# pylint: disable=W0223
class Visualizer:
    def __init__ (self, mesh_obj_filename, window_size, face_detector, head_pose_estimator):
        self.face_detector = face_detector
        self.head_pose_estimator = head_pose_estimator
        mesh = trimesh.load(mesh_obj_filename)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        self.size = window_size
        self.head = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        self.scene = pyrender.Scene(bg_color=(0,0,0,0), ambient_light=(255, 255, 255))
        self.scene.add_node(self.head)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
        camera_pos = np.eye(4)
        camera_pos[2][3] = .5
        self.scene.add(camera, pose=camera_pos)
        self.cap = cv2.VideoCapture(0)

    def run(self):
        v = pyrender.Viewer(self.scene, viewport_size=self.size, use_raymond_lighting=True, run_in_thread=True)

        while True:
            _, img = self.cap.read()
            h, w, _ = img.shape
            detections = self.face_detector(img)
            if len(detections) == 1:
                face = detections[0]
                crop = img[face.ymin:face.ymax, face.xmin:face.xmax]
                pose = self.head_pose_estimator(crop)
                r = euler_to_rotation_matrix(pose["yaw"], pose["pitch"], pose["roll"], angles=True)
                v.render_lock.acquire()
                self.scene.set_pose(self.head, r)
                v.render_lock.release()

