import logging
import os

import cv2
import numpy as np


log = logging.getLogger('Global log')


def tensor2image(tensor):
    image = tensor * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1,2,0)[:,:,[2,1,0]]
    return image.astype(np.uint8).copy()


def write_obj(obj_name,
              vertices,
              faces,
              colors=None,
              texture=None,
              uvcoords=None,
              uvfaces=None,
              inverse_face_order=False,
              normal_map=None,
              ):
    ''' Save 3D face model with texture.
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        faces: shape = (ntri, 3)
        texture: shape = (uv_size, uv_size, 3)
        uvcoords: shape = (nver, 2) max value<=1
    '''
    if os.path.splitext(obj_name)[-1] != '.obj':
        obj_name = obj_name + '.obj'
    # mtl_name = obj_name.replace('.obj', '.mtl')
    # texture_name = obj_name.replace('.obj', '.png')
    # material_name = 'FaceTexture'

    faces = faces.copy()
    # mesh lab start with 1, python/c++ start from 0
    faces += 1
    # if inverse_face_order:
    #     faces = faces[:, [2, 1, 0]]
    #     if uvfaces is not None:
    #         uvfaces = uvfaces[:, [2, 1, 0]]

    # write obj
    log.info(f"Saving resulted .obj file: {obj_name}")
    with open(obj_name, 'w') as f:
        # first line: write mtlib(material library)
        # f.write('# %s\n' % os.path.basename(obj_name))
        # f.write('#\n')
        # f.write('\n')
        # if texture is not None:
        #     f.write('mtllib %s\n\n' % os.path.basename(mtl_name))

        # write vertices
        if colors is None:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        else:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1],
                    vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

        # # write uv coords
        if texture is None:
            for i in range(faces.shape[0]):
                f.write('f {} {} {}\n'.format(faces[i, 2], faces[i, 1], faces[i, 0]))
        # else:
        #     for i in range(uvcoords.shape[0]):
        #         f.write('vt {} {}\n'.format(uvcoords[i,0], uvcoords[i,1]))
        #     f.write('usemtl %s\n' % material_name)
        #     # write f: ver ind/ uv ind
        #     uvfaces = uvfaces + 1
        #     for i in range(faces.shape[0]):
        #         f.write('f {}/{} {}/{} {}/{}\n'.format(
        #             #  faces[i, 2], uvfaces[i, 2],
        #             #  faces[i, 1], uvfaces[i, 1],
        #             #  faces[i, 0], uvfaces[i, 0]
        #             faces[i, 0], uvfaces[i, 0],
        #             faces[i, 1], uvfaces[i, 1],
        #             faces[i, 2], uvfaces[i, 2]
        #         )
        #         )
        #     # write mtl
        #     with open(mtl_name, 'w') as f:
        #         f.write('newmtl %s\n' % material_name)
        #         s = 'map_Kd {}\n'.format(os.path.basename(texture_name)) # map to image
        #         f.write(s)

        #         if normal_map is not None:
        #             name, _ = os.path.splitext(obj_name)
        #             normal_name = f'{name}_normals.png'
        #             f.write(f'disp {normal_name}')
        #             # out_normal_map = normal_map / (np.linalg.norm(
        #             #     normal_map, axis=-1, keepdims=True) + 1e-9)
        #             # out_normal_map = (out_normal_map + 1) * 0.5

        #             cv2.imwrite(
        #                 normal_name,
        #                 # (out_normal_map * 255).astype(np.uint8)[:, :, ::-1]
        #                 normal_map
        #             )
        #     cv2.imwrite(texture_name, texture)


def save_obj(obj_filename, opdict, template_path):
    '''
    vertices: [nv, 3], tensor
    texture: [3, h, w], tensor
    '''
    i = 0
    vertices = opdict['verts'][i]
    verts, uvcoords, faces, uvfaces= load_obj(template_path)
    faces = faces[None, ...][0]
    # texture = tensor2image(opdict['uv_texture_gt'][i])
    # uvcoords = self.render.raw_uvcoords[0]
    # uvfaces = self.render.uvfaces[0]
    # save coarse mesh, with texture and normal map
    #normal_map = tensor2image(opdict['uv_detail_normals'][i] * 0.5 + 0.5)
    write_obj(obj_filename, vertices, faces)
                    # texture=texture,
                    # uvcoords=uvcoords,
                    # uvfaces=uvfaces,
                    # normal_map=normal_map)
    # # upsample mesh, save detailed mesh
    # texture = texture[:,:,[2,1,0]]
    # normals = opdict['normals'][i].cpu().numpy()
    # displacement_map = opdict['displacement_map'][i].cpu().numpy().squeeze()
    # dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals,
    #   faces, displacement_map, texture, self.dense_template)
    # util.write_obj(filename.replace('.obj', '_detail.obj'),
    #                 dense_vertices,
    #                 dense_faces,
    #                 colors = dense_colors,
    #                 inverse_face_order=True)


def load_obj(obj_filename):
    """
    Ref:
    https://github.com/facebookresearch/pytorch3d/blob/25c065e9dafa90163e7cec873dbb324a637c68b7/pytorch3d/io/obj_io.py
    Load a mesh from a file-like object.
    """
    with open(obj_filename, 'r') as f:
        lines = [line.strip() for line in f]

    verts, uvcoords = [], []
    faces, uv_faces = [], []
    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            uvcoords.append(tx)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            face = tokens[1:]
            face_list = [f.split("/") for f in face]
            for vert_props in face_list:
                # Vertex index.
                faces.append(int(vert_props[0]))
                if len(vert_props) > 1:
                    if vert_props[1] != "":
                        # Texture index is present e.g. f 4/1/1.
                        uv_faces.append(int(vert_props[1]))

    faces = np.array(faces).reshape(-1, 3) - 1
    uv_faces = np.array(uv_faces).reshape(-1, 3) - 1
    return (
        np.array(verts),
        np.array(uvcoords),
        faces,
        uv_faces
    )