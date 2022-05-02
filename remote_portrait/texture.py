import numpy as np
import torch
import torch.nn.functional as F

import cv2

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes


class Rasterizer:
    ## TODO: add support for rendering non-squared images, since pytorc3d supports this now
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=256):
        """
        use fixed raster_settings for rendering faces
        """
        self.raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin':  None,
            'perspective_correct': False,
        }

    def __call__(self, vertices, faces, attributes=None, h=None, w=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]
        if h is None and w is None:
            image_size = self.raster_settings['image_size']
        else:
            image_size = [h, w]
            if h > w:
                fixed_vertices[..., 1] = fixed_vertices[..., 1]*h/w
            else:
                fixed_vertices[..., 0] = fixed_vertices[..., 0]*w/h

        meshes_screen = Meshes(verts=fixed_vertices, faces=faces)
        pix_to_face, _, bary_coords, _ = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=self.raster_settings['blur_radius'],
            faces_per_pixel=self.raster_settings['faces_per_pixel'],
            bin_size=self.raster_settings['bin_size'],
            max_faces_per_bin=self.raster_settings['max_faces_per_bin'],
            perspective_correct=self.raster_settings['perspective_correct'],
        )
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
        pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)

        return pixel_vals


class Texture:
    def __init__(self, head_template_obj, fixed_uv_displacement, uv_face_eye_mask):
        self.fixed_uv_displacement = fixed_uv_displacement[None,None,:,:]
        self.uv_face_eye_mask = uv_face_eye_mask[np.newaxis, np.newaxis]
        self.rasterizer = Rasterizer()
        self.dense_faces = Texture.generate_triangles(256, 256)

        _, templ_faces, templ_aux = head_template_obj
        templ_uvcoords = templ_aux.verts_uvs[None, ...]  # (N, V, 2)
        self.raw_uvcoords = templ_uvcoords.clone()

        templ_uvcoords = torch.cat([templ_uvcoords, templ_uvcoords[:,:,0:1] * 0. + 1.], -1) #[bz, ntv, 3]
        templ_uvcoords = templ_uvcoords * 2 - 1
        templ_uvcoords[...,1] = -templ_uvcoords[...,1]
        self.templ_uvcoords = templ_uvcoords
        self.templ_uvfaces = templ_faces.textures_idx[None, ...]  # (N, F, 3)
        self.templ_faces = templ_faces.verts_idx[None,...]

    def __call__(self, face_image, albedo, uv_z, verts, trans_verts, light):
        #preprocess images
        face_image = cv2.normalize(face_image, None, alpha=0 , beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = face_image[np.newaxis].transpose(0,3,1,2)
        self.uv_face_eye_mask = cv2.normalize(self.uv_face_eye_mask, None, alpha=0,
            beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        normals = Texture.normals(verts[0], self.templ_faces)
        uv_detail_normals = self.displacement2normal(uv_z, verts, normals, self.uv_face_eye_mask,
            self.templ_faces, self.templ_uvcoords, self.templ_uvfaces)
        uv_shading = Texture.add_sh_light(uv_detail_normals, light)
        uv_texture = np.multiply(albedo, np.float32(uv_shading))

        uv_pverts = self.world2uv(trans_verts, self.templ_faces, self.templ_uvcoords, self.templ_uvfaces)
        uv_gt = F.grid_sample(torch.from_numpy(face_image), uv_pverts.permute(0,2,3,1)[:,:,:,:2], mode='bilinear')

        ## TODO: poisson blending should give better-looking results
        uv_texture_gt = np.multiply(uv_gt[:,:3,:,:], self.uv_face_eye_mask) \
            + np.multiply(uv_texture[:,:3,:,:], (1 - self.uv_face_eye_mask))

        vis_texture = cv2.cvtColor(uv_texture_gt.numpy().transpose(0, 2, 3, 1)[0], cv2.COLOR_RGB2BGR)
        cv2.imshow("uv_texture", vis_texture)

        return vis_texture * 255, self.raw_uvcoords, self.templ_uvfaces

    @staticmethod
    def get_face_vertices(vertices, faces):
        """
        :param vertices: [batch size, number of vertices, 3]
        :param faces: [batch size, number of faces, 3]
        :return: [batch size, number of faces, 3, 3]
        """
        if len(vertices.shape) == 3:
            vertices = vertices[0]

        return vertices[faces]

    def world2uv(self, vertices, faces, uvcoords, uvfaces):
        '''
        warp vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        '''
        face_vertices = Texture.get_face_vertices(vertices, faces)
        uv_vertices = self.rasterizer(uvcoords,uvfaces, torch.from_numpy(face_vertices))[:, :3] # remove
        return uv_vertices

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals, uv_face_eye_mask, faces, uvcoords, uvfaces):
        ''' Convert displacement map into detail normal map
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.world2uv(coarse_verts, faces, uvcoords, uvfaces)
        uv_coarse_normals = self.world2uv(coarse_normals, faces, uvcoords, uvfaces)

        uv_z = np.multiply(uv_z, uv_face_eye_mask)
        uv_detail_vertices = uv_coarse_vertices.cpu().detach().numpy() + \
            np.multiply(uv_z, uv_coarse_normals.cpu().detach().numpy()) + \
            np.multiply(self.fixed_uv_displacement, uv_coarse_normals.cpu().detach().numpy())
        dense_vertices = np.transpose(uv_detail_vertices, (0,2,3,1)).reshape([batch_size, -1, 3])
        uv_detail_normals = self.normals(dense_vertices[0], self.dense_faces)
        uv_detail_normals = np.transpose(uv_detail_normals.reshape(
            [batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]), (0,3,1,2))
        uv_detail_normals = np.multiply(uv_detail_normals, uv_face_eye_mask) + \
            np.multiply(uv_coarse_normals.numpy(), (1 - uv_face_eye_mask))
        return uv_detail_normals

    @staticmethod
    def add_sh_light(texture, sh_coeff):
        constant_factor = np.array([1 / np.sqrt(4 * np.pi),
                                    ((2 * np.pi)/3) * (np.sqrt(3 / (4 * np.pi))),
                                    ((2 * np.pi) / 3) * (np.sqrt(3 / (4 * np.pi))),
                                    ((2 * np.pi) / 3)*(np.sqrt(3 / (4 * np.pi))),
                                    (np.pi / 4) * 3 * (np.sqrt(5 / (12 * np.pi))),
                                    (np.pi / 4) * 3 * (np.sqrt(5 / (12 * np.pi))),
                                    (np.pi / 4) * 3 * (np.sqrt(5 / (12 * np.pi))),
                                    (np.pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * np.pi))),
                                    (np.pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * np.pi)))])
        sh = np.stack([
            texture[:, 0] * 0.+1.,
            texture[:, 0],
            texture[:, 1],
            texture[:, 2],
            texture[:, 0] * texture[:, 1],
            texture[:, 0] * texture[:, 2],
            texture[:, 1] * texture[:, 2],
            texture[:, 0]**2 - texture[:, 1]**2,
            3 * (texture[:, 2]**2) - 1
            ],
            1) # [bz, 9, h, w]
        sh = np.multiply(sh, constant_factor[None,:,None,None])
        shading = np.sum(np.multiply(sh_coeff[:,:,:,None,None], sh[:,:,None,:,:]), 1)
        return shading

    @staticmethod
    def normals(vertices, faces):
        """
        :param vertices: [batch size, number of vertices, 3]
        :param faces: [batch size, number of faces, 3]
        :return: [batch size, number of vertices, 3]
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        normals = np.zeros(vertices.shape)
        np.add.at(normals, faces[:, 0], np.cross(v1 - v0, v2 - v0))
        np.add.at(normals, faces[:, 1], np.cross(v2 - v1, v0 - v1))
        np.add.at(normals, faces[:, 2], np.cross(v0 - v2, v1 - v2))

        nonzero = ~np.isclose(normals, [0., 0., 0.])[:,1] # create row mask
        normals[nonzero] /= np.linalg.norm(normals, axis=1, keepdims=True)[nonzero]

        return normals  #normals[~np.isnan(normals).any(axis=1)]

    @staticmethod
    def generate_triangles(h, w, margin_x=2, margin_y=5):
        # quad layout:
        # 0 1 ... w-1
        # w w+1
        #.
        # w*h
        triangles = []
        for x in range(margin_x, w-1-margin_x):
            for y in range(margin_y, h-1-margin_y):
                triangle0 = [y*w + x, y*w + x + 1, (y+1)*w + x]
                triangle1 = [y*w + x + 1, (y+1)*w + x + 1, (y+1)*w + x]
                triangles.append(triangle0)
                triangles.append(triangle1)
        triangles = np.array(triangles)
        triangles = triangles[:,[0,2,1]]
        return triangles
