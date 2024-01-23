from torch.autograd import Function

from .forward import forward
from .backward import backward


class TorchForward(Function):
    @staticmethod
    def forward(ctx, volume, geom):
        sinogram = forward(volume, geom)
        ctx.geom = geom
        return sinogram

    @staticmethod
    def backward(ctx, grad_vol):
        grad = backward(grad_vol, ctx.geom)
        return grad, None, None, None, None, None, None


class TorchBackprojection(Function):
    @staticmethod
    def forward(ctx, sino, geom):
        ctx.geom = geom
        volume = backward(sino, geom)

        return volume

    @staticmethod
    def backward(ctx, grad_sino):
        grad = forward(grad_sino, ctx.geom)
        return grad, None, None, None, None, None, None
