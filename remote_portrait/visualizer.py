import logging
import threading

import cv2
import numpy as np
import pyrender
import trimesh

from .utils import adjust_rect, square_crop_resize

log = logging.getLogger('Global log')


class AnimationThread(threading.Thread):
    def __init__(self, render_lock, scene, head, mesh, parameters, face_detector, flame_encoder, flame):
        threading.Thread.__init__(self)
        self.render_lock = render_lock
        self.scene = scene
        self.head = head
        self.mesh = mesh
        self.parameters = parameters
        self.face_detector = face_detector
        self.flame_encoder = flame_encoder
        self.flame = flame
        self.cap = cv2.VideoCapture(0)

    def run(self):
        while True:
            _, img = self.cap.read()
            h, w, _ = img.shape
            detections = self.face_detector(img)

            if len(detections) == 1:
                face = detections[0]
                bottom_left, top_right = adjust_rect(face.xmin, face.ymin,
                face.xmax, face.ymax, 0.15, 0.2)
                cropped_face = square_crop_resize(img, bottom_left, top_right, 224)
                cv2.rectangle(img, bottom_left, top_right, (0, 255, 0))
                parameters = self.flame_encoder(cropped_face)
                # transfer only pose and expression
                self.parameters['pose'] = parameters['pose']
                self.parameters['exp'] = parameters['exp']
                result_dict = self.flame(self.parameters)
                # modify meshes
                self.mesh.vertices = result_dict['verts'][0]
                mesh = pyrender.Mesh.from_trimesh(self.mesh)
                self.render_lock.acquire()
                self.head.mesh = mesh
                self.render_lock.release()
            cv2.imshow("img", img)
            cv2.waitKey(1)


# pylint: disable=W0223
class Visualizer:
    def __init__ (self, mesh_obj_filename, window_size, orig_params, face_detector, flame_encoder, flame):
        self.parameters = orig_params
        self.face_detector = face_detector
        self.flame_encoder = flame_encoder
        self.flame = flame

        self.mesh = trimesh.load(mesh_obj_filename, process=False, maintain_order=True)
        mesh = pyrender.Mesh.from_trimesh(self.mesh)

        self.size = window_size
        self.head = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        self.scene = pyrender.Scene(bg_color=(0,0,0,0), ambient_light=(255, 255, 255))
        self.scene.add_node(self.head)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
        camera_pos = np.eye(4)
        camera_pos[2][3] = .5
        self.scene.add(camera, pose=camera_pos)

    def run(self):
        v = pyrender.Viewer(self.scene, viewport_size=self.size, use_raymond_lighting=True, run_in_thread=True)
        animation_thread = AnimationThread(v._render_lock, self.scene, self.head, self.mesh, self.parameters,
            self.face_detector, self.flame_encoder, self.flame)
        animation_thread.start()
