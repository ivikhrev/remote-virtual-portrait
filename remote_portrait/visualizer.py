import ctypes
import sys

import pyglet
from pyglet.gl import *
from pywavefront import visualization
import pywavefront

def visualize(obj_file_name):
    meshes = pywavefront.Wavefront(obj_file_name)
    global pos
    global rot_x, rot_y, pos_z, zoom, outName
    rotation = 0
    rot_x = 0
    rot_y = 0
    zoom = 0

    pos = [0, 0, -1]

    out_name = "test.png"
    config = Config(sample_buffers=1, samples=8)
    window = pyglet.window.Window(height=800, width=600, config=config)
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
        global rot_x, rot_y, outName
        lightfv = ctypes.c_float * 4
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

        # pyglet.image.get_buffer_manager().get_color_buffer().save(out_name)

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

    pyglet.app.run()

