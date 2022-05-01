import logging

import pyglet
from pyglet import gl
import pywavefront
from pywavefront import visualization  # pylint: disable=W0611


log = logging.getLogger('Global log')

pywavefront.configure_logging(
    logging.DEBUG,
    formatter=logging.Formatter('[ %(levelname)s ] %(message)s')
)


# pylint: disable=W0223
class Visualizer(pyglet.window.Window):
    def __init__ (self, width, height, mesh_obj_filename, show=True):
        super().__init__(width=width, height=height, visible=show, fullscreen=False)
        self.show = show
        self.x, self.y, self.z = 0, 0, -1
        self.rot_x, self.rot_y, self.rot_z = 0, -15, 0
        self.res_img_name = mesh_obj_filename.split('.')[0] + '.png'
        self.is_running = True
        self.meshes = pywavefront.Wavefront(mesh_obj_filename)

    def on_resize(self, width, height):  # pylint: disable=R0201
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluPerspective(30., float(width)/height, 0.01, 100.)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        return True

    def on_draw(self):
        self.render()

    def render(self):
        self.clear()
        gl.glLoadIdentity()
        gl.glEnable(gl.GL_LIGHTING)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat*4)(0, 0, 2, 0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (gl.GLfloat*4)(0, 0, 0, 1))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (gl.GLfloat*4)(1, 1, 1, 1))
        gl.glEnable(gl.GL_LIGHT0)

        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glShadeModel(gl.GL_SMOOTH)

        gl.glTranslatef(self.x, self.y, self.z)
        gl.glRotatef(self.rot_y, 1, 0, 0)
        gl.glRotatef(self.rot_x, 0, 1, 0)
        gl.glRotatef(self.rot_z, 0, 0, 1)
        pywavefront.visualization.draw(self.meshes)

    def on_close(self):
        self.is_running = False

    def on_key_press(self, symbol, modifiers):  # pylint: disable=W0613
        if symbol ==  pyglet.window.key.S:
            pyglet.image.get_buffer_manager().get_color_buffer().save(self.res_img_name)
            log.info(f"Saved image to {self.res_img_name}")
        if symbol == pyglet.window.key.ESCAPE:
            self.is_running = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):  # pylint: disable=W0613
        if buttons & pyglet.window.mouse.LEFT:
            self.rot_y += dy
            self.rot_x += dx

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):  # pylint: disable=W0613
        self.z += 0.1 * scroll_y

    def run(self):
        # to setup image for no show option
        self.on_resize(self.width, self.height)
        while self.is_running:
            pyglet.clock.tick()
            self.render()
            self.flip()
            self.dispatch_events()

            if not self.show:
                for _ in range(1): # skip first frame to make image fully randered
                    self.render()
                    self.flip()
                pyglet.image.get_buffer_manager().get_color_buffer().save(self.res_img_name)
                log.info(f"Saved image to {self.res_img_name}")
                self.on_close()
