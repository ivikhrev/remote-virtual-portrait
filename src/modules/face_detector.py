import face_alignment
import numpy as np

class FAN:
    def __init__(self):
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', flip_input=False)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:,0])
            right = np.max(kpt[:,0])
            top = np.min(kpt[:,1])
            bottom = np.max(kpt[:,1])
            bbox = [(round(left), round(top)), (round(right), round(bottom))]
            return bbox, 'kpt68'

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center