import numpy as np
import logging as log


def log_model_info(model):
    log.info(f"Model name: {model.get_name()}")
    log.info("Inputs:")
    for input_ in model.inputs:
        log.info(f"\t{input_.get_any_name()} : shape {input_.shape}")
    log.info("Outputs:")
    for output_ in model.outputs:
        log.info(f"\t{output_.get_any_name()} : shape {output_.shape}")


def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.copy().reshape((-1, 1, 3))
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = np.concatenate([X_trans, X[:,:,2:]], axis=2)
    shape = X_trans.shape
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn
