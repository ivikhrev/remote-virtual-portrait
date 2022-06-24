from time import perf_counter
import os
import logging

import cv2
import numpy as np
import pyvirtualcam

from .ui import *
from ..modules.images_capture import OpenError, open_images_capture
from ..modules.visualizer import Visualizer
from ..modules.performance_metrics import PerformanceMetrics
from ..modules.reconstruction_pipeline import reconstruct
from ..modules.utils import resize_image_letterbox

log = logging.getLogger('Global log')

class Thread(QThread):
    render = pyqtSignal(QPixmap)
    camera = pyqtSignal(QPixmap)
    do_reconstruction_pipeline = False

    def __init__(self, stub_webcam_image, stub_vis_image, user_config):
        QThread.__init__(self)
        print(os.path.abspath(__file__))
        self.pipeline_metrics = PerformanceMetrics()
        self.user_config = user_config
        self.stub_webcam_image = stub_webcam_image
        self.stub_vis_image =  stub_vis_image
        self.keep_running = True
        self.i = 0

    def run(self):
        while self.keep_running:
            if not self.do_reconstruction_pipeline:
                self.idle_pipeline()
            else:
                self.reconstruction_pipeline()

    def pass_image(self, image_path):
        if image_path is not None and image_path != "":
            print(image_path)
            self.user_config['test_input'] = image_path
            self.do_reconstruction_pipeline = True

    def idle_pipeline(self):
        image = np.zeros((600, 800, 4), np.uint8)
        cv2.putText(image, "." * self.i, (350, 550), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 0, 255), 10)
        self.i = (self.i + 1) % 4
        webcam_pixmap = self.join_pixmap(
            self.convert_cv_qt(image, self.stub_webcam_image.width(), self.stub_webcam_image.height()), self.stub_webcam_image)
        vis_pixmap = self.join_pixmap(
            self.convert_cv_qt(image, self.stub_vis_image.width(), self.stub_vis_image.height()), self.stub_vis_image)
        self.camera.emit(webcam_pixmap)
        self.render.emit(vis_pixmap)
        cv2.waitKey(300)

    def reconstruction_pipeline(self):
        try:
            output_name, parameters, fast_face_detector, flame_encoder, flame = reconstruct(self.user_config)
            self.visualizer = Visualizer(output_name, self.user_config["visualizer_size"], parameters, fast_face_detector, flame_encoder, flame)
            self.cap = open_images_capture(0, True)
            try:
                cam = pyvirtualcam.Camera(width=1280,
                    height=720, fps=30)
            except RuntimeError:
                print("Can't open virtual camera")
                cam = None
            while self.keep_running and self.do_reconstruction_pipeline:
                start_time = perf_counter()
                img = self.cap.read()
                render_image, bottom_left, top_right = self.visualizer.run(img)
                if bottom_left is not None and top_right is not None:
                        cv2.rectangle(img, bottom_left, top_right, (255, 0, 0), 2)
                camera_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #self.convert_cv_qt(img, img.shape[0], img.shape[1])# c
                qtImage = QImage(camera_image.data, camera_image.shape[1], camera_image.shape[0],
                    QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qtImage)
                self.camera.emit(pixmap)
                render_image = cv2.cvtColor(render_image, cv2.COLOR_RGBA2RGB)
                if cam is not None:
                    cam.send(resize_image_letterbox(render_image, (1280, 720)))
                    cam.sleep_until_next_frame()
                self.pipeline_metrics.update(start_time, render_image)
                qtImage = QImage(render_image.data, render_image.shape[1], render_image.shape[0],
                    QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qtImage)
                self.render.emit(pixmap)
                cv2.waitKey(1)
            self.visualizer.delete_renderer()
            self.cap.release()
            log.info("Face detection")
            self.visualizer.fd_metrics.log_total()
            log.info("Encoder")
            self.visualizer.flame_encoder_metrics.log_total()
            log.info("flame")
            self.visualizer.flame_metrics.log_total()
            log.info("Rendering")
            self.visualizer.render_metrics.log_total()
            log.info("General")
            self.pipeline_metrics.log_total()
        except OpenError:
            print("Can't open web-camera")
            self.do_reconstruction_pipeline = False


    @staticmethod
    def join_pixmap(p1, p2, mode=QPainter.CompositionMode.CompositionMode_Xor):
        s = p1.size().expandedTo(p2.size())
        result = QPixmap(s)
        result.fill(Qt.transparent)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawPixmap(QPoint(), p1)
        painter.setCompositionMode(mode)
        painter.drawPixmap(result.rect(), p2, p2.rect())
        painter.end()
        return result

    @staticmethod
    def convert_cv_qt(cv_img, dest_w, dest_h):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            cv_img.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
        p = convert_to_Qt_format.scaled(
            dest_w, dest_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QPixmap.fromImage(p)

    def stop(self):
        self.keep_running = False
        self.exit()
        print("thread stopped")


class Execution(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_ui()

    def init_ui(self):
        self.win = MainWindow()
        self.win.show()

        self.thread = Thread(self.win.webcameraLabel.pixmap().copy(), self.win.visualizerLabel.pixmap().copy(), self.config)
        self.thread.camera.connect(self.show_webcamera)
        self.thread.render.connect(self.show_visualizer)
        self.thread.start()
        print("thread started")
        self.win.closeEvent = self.closeEvent
        self.win.btnSelectImage.clicked.connect(self.select_image)

    @pyqtSlot(QPixmap)
    def show_webcamera(self, image):
        self.win.webcameraLabel.setPixmap(image)
        self.win.update()

    @pyqtSlot(QPixmap)
    def show_visualizer(self, image):
        self.win.visualizerLabel.setPixmap(image)
        self.win.update()

    def select_image(self):
        self.thread.do_reconstruction_pipeline = False
        file_name = QFileDialog.getOpenFileName(self,
            'Open File',"",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)")[0]
        self.thread.pass_image(file_name)

    def close_event(self, event):
        print("thread stop")
        self.thread.stop()
        event.accept()