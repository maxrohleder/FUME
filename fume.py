import torch
import torch.nn as nn
import numpy as np
import fume_torch_lib  # name defined in setup.py as cuda_extension


def calculate_fundamental_matrix(P_src, P_dst):
    """
    Returns the fundamental matrix F which maps a coordinate x1 in projection P_src onto a line l2 in projection P_dst
    l2 = F21 @ x1
    @param P_src: 3x4 projection matrix
    @param P_dst: 3x4 projection matrix
    @return: Fundamental Matrix F in shape (3, 3)
    """
    assert P_src.shape == (3, 4) and P_dst.shape == (3, 4)

    # compute the null space of the Projection matrix == camera center
    C1 = -np.linalg.inv(P_src[:3, :3]) @ P_src[:3, 3]

    # compute epipole == projection of camera center on target view plane
    e = P_dst @ np.pad(C1, (0, 1), constant_values=1)

    # construct the fundamental matrix
    return np.cross(e, np.eye(e.shape[0]) * -1) @ P_dst @ np.linalg.pinv(P_src)


class FumeImageTranslation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, F, Finv):
        '''
        Translates an input view tensor in shape (B, C, H, W) onto an epipolar similar 
        shaped image using the geometry expressed through the fundamental matrix F in 
        shape (3, 3). The inverse operation is saved for the backward pass.
        '''
        ctx.save_for_backward(Finv)
        return fume_torch_lib.translate(input, F)

    @staticmethod
    def backward(ctx, grad):
        '''
        This gradient shaped (B, C, H, W) is mapped onto the similar shaped input.
        '''
        Finv = ctx.saved_tensors[0]
        return fume_torch_lib.translate(grad, Finv)


class Fume3dLayer(nn.Module):
    def __init__(self):
        super(Fume3dLayer, self).__init__()


    def forward(self, view1, F21, F12):
        assert view1.ndim == 4, "Input must have shape (B, C, H, W)"
        # returns view1 mapped into the projective transform P2
        return FumeImageTranslation.apply(view1, F21, F12)
