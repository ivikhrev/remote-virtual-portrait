import ctypes
import logging

import pyglet
from pyglet.gl import *
import pywavefront
from pywavefront import visualization

log = logging.getLogger('Global log')

pywavefront.configure_logging(
    logging.DEBUG,
    formatter=logging.Formatter('[ %(levelname)s ] %(message)s')
)


class Visualizer(pyglet.window.Window):
    def __init__ (self, width, height, mesh_obj_filename, show=True):
        super(Visualizer, self).__init__(width=width, height=height, visible=show, fullscreen = False)
        self.show = show
        self.x, self.y, self.z = 0, 0, -1
        self.rot_x, self.rot_y = 0, 0
        self.res_img_name = mesh_obj_filename.split('.')[0] + '_img.png'
        self.is_running = True
        self.meshes = pywavefront.Wavefront(mesh_obj_filename)

    def on_resize(self, width, height):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(30., float(width)/height, 0.01, 100.)
        glMatrixMode(GL_MODELVIEW)
        return True

    def on_draw(self):
        self.render()

    def render(self):
        self.clear()
        glLoadIdentity()
        glEnable(GL_LIGHTING)
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat*4)(5, 5, 5, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat*4)(0, 0, 0, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat*4)(1, 1, 1, 1))
        glEnable(GL_LIGHT0)

        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)

        glTranslatef(self.x, self.y, self.z)
        glRotatef(self.rot_y, 1, 0, 0)
        glRotatef(self.rot_x, 0, 1, 0)
        pywavefront.visualization.draw(self.meshes)

    def on_close(self):
        self.is_running = False

    def on_key_press(self, symbol, modifiers):
        if symbol ==  pyglet.window.key.S:
            pyglet.image.get_buffer_manager().get_color_buffer().save(self.res_img_name)
        if symbol == pyglet.window.key.ESCAPE:
            self.alive = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & pyglet.window.mouse.LEFT:
            self.rot_y += dy
            self.rot_x += dx

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
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
                self.on_close()


def visualize(obj_file_name, show=True):
    meshes = pywavefront.Wavefront(obj_file_name)
    global pos
    global rot_x, rot_y, pos_z, zoom, outName
    rotation = 0
    rot_x = 0
    rot_y = 0
    zoom = 0

    pos = [0, 0, -1]

    out_name = "test.png"
    #config = Config(sample_buffers=1, samples=8)
    print(show)
    window = pyglet.window.Window(height=800, width=600, visible=show) #, config=config)
    lightfv = ctypes.c_float * 4

    @window.event
    def on_resize(width, height):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(30., float(width)/height, 0.01, 100.)
        glMatrixMode(GL_MODELVIEW)
        return True


    @window.event
    def on_draw():
        global rot_x, rot_y, out_name
        window.clear()
        glLoadIdentity()
        glEnable(GL_LIGHTING)
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat*4)(5, 5, 5, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat*4)(0, 0, 0, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat*4)(1, 1, 1, 1))
        glEnable(GL_LIGHT0)

        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)

        glTranslatef(*pos)
        glRotatef(rot_y, 1, 0, 0)
        glRotatef(rot_x, 0, 1, 0)
        # glOrtho(self.camera.x, self.camera.x2, self.camera.y, self.camera.y2, -1, 1)
        visualization.draw(meshes)

    @window.event
    def on_key_press(s, m):
        print(m)
        print(s)
        if s ==  pyglet.window.key.S:
            pyglet.image.get_buffer_manager().get_color_buffer().save(out_name)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        global rot_x, rot_y
        if buttons & pyglet.window.mouse.LEFT:
            rot_y += dy
            rot_x += dx

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        global pos
        pos[2] += 0.1 * scroll_y

    if show:
        pyglet.app.run()
    else:
        on_draw()
        on_resize(800, 600)
        on_key_press(pyglet.window.key.S, None)

