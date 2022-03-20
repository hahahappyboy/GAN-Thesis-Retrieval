
from torch.autograd import Variable
from torch.autograd import grad
import torch.autograd as autograd
import torch.nn as nn
import torch
import numpy as np
def R1Penalty(real_img, f):
    # gradient penalty
    reals = Variable(real_img, requires_grad=True)
    real_logit = f(reals)
    apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]))
    undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]))

    real_logit = apply_loss_scaling(torch.sum(real_logit))
    real_grads = grad(real_logit, reals, grad_outputs=torch.ones(real_logit.size()), create_graph=True)[0].view(reals.size(0), -1)
    real_grads = undo_loss_scaling(real_grads)
    r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
    return r1_penalty