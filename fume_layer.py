import torch
import torch.nn as nn

import fume_cuda  # name defined in setup.py as cuda_extension

def calculate_fundamental_matrix(P_src, P_dst):
    """
    TODO: implement fundamental matrix calculation
    """
    pass


class FumeImageTranslation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, F, Finv):
        '''
        Translates an input view tensor in shape (B, C, H, W) onto an epipolar similar 
        shaped image using the geometry expressed through the fundamental matrix F in 
        shape (3, 3). The inverse operation is saved for the backward pass.
        '''
        ctx.save_for_backward(Finv)
        return fume_cuda.translate(input, F)

    @staticmethod
    def backward(ctx, grad):
        '''
        This gradient in shape (B, C, H, W) is mapped onto the similar shaped input.
        '''
        Finv = ctx.saved_tensors[0]
        return fume_cuda.translate(grad, Finv)



class Fume3d(nn.Module):
    def __init__(self, P1, P2):
        super(Fume3d, self).__init__()
        '''
        Given two projection matrices P1, P2 in shape (3, 4) this operator calculates epipolar images.
        '''
        self.F21 = calculate_fundamental_matrix(P_src=P1, P_dst=P2)
        self.F12 = calculate_fundamental_matrix(P_src=P2, P_dst=P1)


    def forward(self, input):
        assert input.ndim == 4, "Input must have shape (B, C, H, W)"
        return FumeImageTranslation.apply(input, self.F21, self.F12)
