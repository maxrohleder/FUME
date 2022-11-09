import torch
import torch.nn as nn
import numpy as np
import fume_torch_lib  # name defined in setup.py as cuda_extension


def calculate_fundamental_matrix(P_src: np.ndarray, P_dst: np.ndarray) -> np.ndarray:
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
    def forward(ctx, input, F, Finv, downsampled_factor):
        '''
        Translates an input view tensor in shape (B, C, H, W) onto an epipolar similar 
        shaped image using the geometry expressed through the fundamental matrix F in 
        shape (3, 3). The inverse operation is saved for the backward pass.
        '''
        ctx.save_for_backward(Finv, downsampled_factor)
        return fume_torch_lib.translate(input, F, downsampled_factor)

    @staticmethod
    def backward(ctx, grad):
        '''
        This gradient shaped (B, C, H, W) is mapped onto the similar shaped input.
        '''
        Finv, downsampled_factor = ctx.saved_tensors
        return fume_torch_lib.translate(grad, Finv, downsampled_factor), None, None, None


class Fume3dLayer(nn.Module):
    def __init__(self):
        super(Fume3dLayer, self).__init__()

    def forward(self, view1, F21, F12, downsampled_factor=None):
        """
        Calculate Epipolar View Line Image
        @param view1: input image
        @param F21: Fundamental Matrix mapping a point in the output `view2` onto a line in the input `view1`
        @param F12: Reverse Mapping; a point on view1 onto a line in view2. Used in the backward pass
        @param downsampled_factor:
        @return: view2 epipolar line image in shape `view1.shape`
        """
        assert view1.ndim == 4, "Input must have shape (B, C, H, W)"
        if downsampled_factor is None:
            downsampled_factor = torch.ones(view1.size(0), dtype=view1.dtype, device='cuda', requires_grad=False)
        # returns a perspective mapping of all pixels in view1 onto view2
        return FumeImageTranslation.apply(view1, F21, F12, downsampled_factor)
