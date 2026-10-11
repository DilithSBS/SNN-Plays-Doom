import torch


class SpikeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, membrane, threshold):

        ctx.save_for_backward(membrane)
        ctx.threshold = threshold

        # Hard spike in the forward pass
        spike = (membrane >= threshold).float()

        return spike

    @staticmethod
    def backward(ctx, grad_output):

        membrane, = ctx.saved_tensors
        threshold = ctx.threshold

        # Surrogate gradient
        distance = membrane - threshold

        surrogate_gradient = (
            1.0 /
            (1.0 + torch.abs(distance)) ** 2
        )

        grad_membrane = (
            grad_output *
            surrogate_gradient
        )

        return grad_membrane, None