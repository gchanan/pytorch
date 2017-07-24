import torch
from torch.autograd.function import Function, InplaceFunction
from torch._thnn import type2backend
from torch.autograd.variable import Variable

from . import _all_functions


class PReLU(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx._backend = type2backend[type(input)]
        output = input.new()
        ctx.num_parameters = weight.numel()
        if ctx.num_parameters == 1:
            # num_parameters == 0 is used indicate that a single weight is shared among all input channels.
            ctx.num_parameters = 0
        ctx._backend.PReLU_updateOutput(
            ctx._backend.library_state,
            input,
            output,
            weight,
            ctx.num_parameters
        )
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_variables
        # alternatively, we could recalculate _backend, num_parameters
        return PReLUBackward.apply(input, weight, grad_output, ctx._backend, ctx.num_parameters)


class PReLUBackward(Function):
    @staticmethod
    def forward(ctx, input, weight, grad_output, backend, num_parameters):
        ctx.save_for_backward(input, weight, grad_output)
        ctx.num_parameters = num_parameters
        grad_input = input.new()
        backend.PReLU_updateGradInput(
            backend.library_state,
            input,
            grad_output,
            grad_input,
            weight,
            num_parameters
        )

        buf = weight.new()
        buf2 = weight.new()
        # TODO: this won't have to be zeroed in the future
        grad_weight = weight.new().resize_as_(weight).zero_()
        backend.PReLU_accGradParameters(
            backend.library_state,
            input,
            grad_output,
            grad_input,
            weight,
            grad_weight,
            buf,
            buf2,
            num_parameters,
            1
        )
        return grad_input, grad_weight

    @staticmethod
    def backward(ctx, ggI, ggW):
        input, weight, gO = ctx.saved_variables
        positive_mask = (input > 0).type_as(ggI)
        nonpositive_mask = (input <= 0).type_as(ggW)
        # Explanation: Let input be i, weight be w, grad_output be gO.
        # f(i, w) = i  if i > 0
        #         = wi if i <= 0
        # df/di * gO  = gO      if i > 0      df/dw * g0 = 0      if i > 0
        #             = g0 * w  if i <= 0                = g0 * i  if i <= 0
        # The rest is taking derivatives of these wrt i, w, gO and summing/expanding properly.
        if ctx.num_parameters == 0:
            # from PReLU.forward: num_parameters == 0 is used indicate that a
            # single weight is shared among all input channels.
            mask = positive_mask + nonpositive_mask * weight.expand_as(input)
            ggO = ggI * mask + ggW.expand_as(gO) * (nonpositive_mask * input)
            return ggW.expand_as(gO) * gO * nonpositive_mask, (ggI * gO * nonpositive_mask).sum(), ggO, None, None
        else:
            # Expand ggW to match size of ggI; a simple expand doesn't work because
            # ggW is the size of the input channel (dim==1 unless there is only 1 dimension).  For example,
            # let ggI be size (3,4,5,6,7) and ggW be size (4).  Then we unsqueeze ggW to be size (4,1,1,1)
            # so the expand succeeds.
            dims_to_unsqueeze = max(input.dim() - 2, 0)
            ggW_expanded = ggW
            for _ in range(dims_to_unsqueeze):
                ggW_expanded = ggW_expanded.unsqueeze(1)
            ggW_expanded = ggW_expanded.expand_as(ggI)

            gI = ggW_expanded * gO * nonpositive_mask

            gW = ggI * gO * nonpositive_mask
            if input.dim() > 1:
                gW = gW.sum(0)
            while gW.dim() > 1:
                gW = gW.sum(1)

            ggO = None
            if gO.requires_grad:
                # expand weight as input as in ggW/ggI above
                weight_expanded = weight
                for _ in range(dims_to_unsqueeze):
                    weight_expanded = weight_expanded.unsqueeze(1)
                weight_expanded = weight_expanded.expand_as(input)

                mask = positive_mask + nonpositive_mask * weight_expanded
                ggO = ggI * mask + ggW_expanded * nonpositive_mask * input
            return gI, gW, ggO, None, None

class BatchNorm2(Function):
    @staticmethod
    def forward(ctx, input, gamma, beta, eps):
        M = input.size(0)
        mu = input.sum(dim=0).div(M)
        sigma2 = (input-mu).pow(2).sum(dim=0).div(M)
        normalized = (input - mu).div((sigma2 + eps).sqrt())
        if gamma is not None:
            ctx.save_for_backward(input, gamma, beta)
        else:
            ctx.save_for_backward(input)
        ctx.eps = eps
        ctx.affine = gamma is not None
        
        if gamma is not None:
            gamma_expanded = gamma
            while len(gamma_expanded.size()) < len(input.size()) - 1:
                gamma_expanded = gamma_expanded.unsqueeze(1)
            beta_expanded = beta
            while len(beta_expanded.size()) < len(input.size()) - 1:
                beta_expanded = beta_expanded.unsqueeze(1)
            return gamma_expanded * normalized + beta_expanded
        else:
            return normalized

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.affine:
            input, gamma, beta = ctx.saved_variables
        else:
            input, = ctx.saved_variables
            gamma = beta = None

        if ctx.affine:
            ret = BatchNorm2Backward.apply(input, gamma, beta, grad_output, ctx.eps)
        else:
            ret = BatchNorm2BackwardNotAffine.apply(input, grad_output, ctx.eps)
            ret = (ret,) + (None, None, None)
        return ret + (None,)


def back(input, gamma, beta, grad_output, eps):
    affine = gamma is not None
    M = input.size(0)
    mu = input.sum(dim=0).div(M).expand_as(input)
    sigma2 = (input - mu).pow(2).sum(dim=0).div(M)

    if not affine:
        gamma_expanded = 1
    else:
        gamma_expanded = gamma
        while len(gamma_expanded.size()) < len(input.size()) - 1:
            gamma_expanded = gamma_expanded.unsqueeze(1)

    first_half = (gamma_expanded / (sigma2 + eps).sqrt()).div(M).expand_as(grad_output)
    second_half = M * grad_output - grad_output.sum(dim=0).expand_as(grad_output) - (input - mu).div((sigma2 + eps).expand_as(grad_output)) * (grad_output * (input - mu)).sum(dim =0).expand_as(grad_output)
    grad_input = first_half * second_half

    if affine:
        grad_beta = grad_output.sum(dim=0)
        while len(grad_beta.size()) > 1:
            grad_beta = grad_beta.sum(dim=1)
    else:
        grad_beta = None
    if affine:
        grad_gamma = ((input - mu) / (sigma2 + eps).sqrt().expand_as(grad_output) * grad_output).sum(dim = 0)
        while len(grad_gamma.size()) > 1:
            grad_gamma =  grad_gamma.sum(dim=1)
    else:
        grad_gamma = None

    return grad_input, grad_gamma, grad_beta


#needs eps too
def backback_no_affine(input, ggI, gO, eps):
    print("something", input, ggI, gO, eps)
    ret = backback_not_affine(input, None, ggI, None, None, gO, eps)
    print("going to return", ret)
    return ret

def backback_not_affine(input, gamma, ggI, ggG, ggB, gO, eps):
    print("in backback_not_affine input", input, "gamma", gamma, "ggI", ggI, "ggG", ggG, "ggB", ggB, "gO", gO, eps)
    M = input.size(0)
    print("expanding as")
    mu = input.sum(dim=0).div(M).expand_as(input)
    print("done expanding as")
    input_min_mu = input - mu
    sigma2 = input_min_mu.pow(2).sum(dim=0).div(M)

    affine = gamma is not None
    if affine:
        gamma_expanded = gamma
        while len(gamma_expanded.size()) < len(input.size()) - 1:
            gamma_expanded = gamma_expanded.unsqueeze(1)

        if ggG is not None:
            ggG_expanded = ggG
            while len(ggG_expanded.size()) < len(input.size()) - 1:
                ggG_expanded = ggG_expanded.unsqueeze(1)
        else:
            ggG_expanded = 0

        if ggB is not None:
            ggB_expanded = ggB
            while len(ggB_expanded.size()) < len(input.size()) - 1:
                ggB_expanded = ggB_expanded.unsqueeze(1)
        else:
            ggB_expanded = 0
            
    else:
        gamma_expanded = 1

    #if ggI is None:
    #    ggI = 0#Variable(input.data.new(1).zero_())

    print("about to caluclate gI")
    # calculate gI
    sigma2_neg_3_2 = (sigma2 + eps).pow(-3 / 2)
    inputmu_sigma2_neg_3_2 = input_min_mu * sigma2_neg_3_2.expand_as(input)
    gOinmu_sum = (gO * input_min_mu).sum(dim=0)


    print("adding up the terms")
    if ggI is not None:
        print("through some of it")
        ggIinmu_sum = (ggI * input_min_mu).sum(dim=0)
        gI_0t = (1 / M * ggI.sum(dim=0) * gO.sum(dim=0) - (gO * ggI).sum(dim=0) +
                 3 / M * (sigma2 + eps).pow(-1) * gOinmu_sum * ggIinmu_sum)
        print("after first term")
        gI_1t = 1 / M * inputmu_sigma2_neg_3_2 * (gI_0t.expand_as(input))
        gI_2t = 1 / M * (ggIinmu_sum * sigma2_neg_3_2).expand_as(gO) * (1 / M * gO.sum(dim=0).expand_as(gO) - gO)
        print("after 3rd term", gI_2t, gamma_expanded)
        gI_3t = 1 / M * (gOinmu_sum * sigma2_neg_3_2).expand_as(ggI) * (1 / M * ggI.sum(dim=0).expand_as(ggI) - ggI)
        print("something")
        gI = gI_1t + gI_2t + gI_3t
        print("done something")
        gI = gamma_expanded * gI
        print("something else", gI)
    else:
        gI = 0

    #print("uh huh", affine, ggG is not None, ggG_expanded is not None)
    #print("second affine check", affine, ggG is not None, ggG_expanded is not None, ggI is not None)
    if affine and ggG is not None:
        # calculate contribution of gamma term to gI
        t0 = gO * (sigma2 + eps).pow(-1 / 2).expand_as(gO)
        t1 = (-1 / M * (sigma2 + eps).pow(-1/2) * (gO).sum(dim=0)).expand_as(input)
        t2 = -1 / M * inputmu_sigma2_neg_3_2 * (gO * input_min_mu).sum(dim=0).expand_as(input)
        gI = gI + ggG_expanded * (t0 + t1 + t2)

    
    # first derivative wrt input
    def first_gI(gO, gamma):
        t0 = (gamma / (sigma2 + eps).sqrt()).div(M).expand_as(gO)
        t1 = (M * gO - gO.sum(dim=0).expand_as(gO) - input_min_mu.div((sigma2 + eps).expand_as(gO)) *
              (gO * input_min_mu).sum(dim=0).expand_as(gO))
        return t0 * t1

    print("gG calculation")
    # calculate gG
    gG = None
    if affine and ggI is not None:
        gG = ggI * first_gI(gO, 1)
        # sum gG over all non-1 dimensions
        gG = gG.sum(dim=0)
        while len(gG.size()) > 1:
            gG = gG.sum(dim=1)

    print("ggB")
    # calculate gB
    gB = Variable(ggB.data.new(ggB.size()).zero_()) if affine and ggB is not None else None

    # calculate ggO
    # contribution of input term
    print("doing firstGi", affine, ggI is not None, ggG is not None, ggB is not None)
    #print("ggG expanded", ggG_expanded)
    ggO = None
    if affine and ggI is not None:
        ggO = first_gI(ggI, gamma_expanded)
    if ggG is not None:
        print("in ggG term")
        ggO_G_term = ggG_expanded * input_min_mu * (sigma2 + eps).pow(-1/2)
        ggO_G_term = ggO_G_term.expand_as(gO) #.expand_as(ggG_expanded)
        print("calculated first term")
        if ggO is None:
            ggO = 0
        ggO = ggO + ggO_G_term
        print("done ggG term")
    if ggB is not None:
        ggO_B_term = ggB_expanded.expand_as(gO)
        if ggO is None:
            ggO = 0
        ggO = ggO + ggO_B_term
        #ggO = ggO + ggO_G_term + ggO_B_term

    print("returning from not_affine gI", gI, "gG", gG, "gB", gB, "ggO", ggO)
    return gI, gG, gB, ggO

class BatchNorm2Backward(Function):
    @staticmethod
    def forward(ctx, input, gamma, beta, grad_output, eps):
        ctx.save_for_backward(input, gamma, grad_output)
        ctx.eps = eps
        return back(input, gamma, beta, grad_output, eps)


    @staticmethod
    def backward(ctx, ggI, ggG, ggB):
        input, gamma, gO = ctx.saved_variables
        return backback_not_affine(input, gamma, ggI, ggG, ggB, gO, ctx.eps) + (None,)


class BatchNorm2BackwardNotAffine(Function):
    @staticmethod
    def forward(ctx, input, grad_output, eps):
        ret = back(input, None, None, grad_output, eps)
        ctx.save_for_backward(input, grad_output)
        ctx.eps = eps
        ret2 = ret[0]
        return ret2

    @staticmethod
    def backward(ctx, ggI):
        input, grad_output = ctx.saved_variables
        ret = backback_not_affine(input, None, ggI, None, None, grad_output, ctx.eps)
        return ret[0], ret[3], None


class RReLU(InplaceFunction):

    def __init__(self, lower, upper, train, inplace=False):
        super(RReLU, self).__init__(inplace)
        self.lower = lower
        self.upper = upper
        self.train = train

    def forward(self, input):
        self._backend = type2backend[type(input)]
        if self.inplace:
            self.mark_dirty(input)
            output = input
        else:
            output = input.new(input.size())
        self.noise = input.new()
        self._backend.RReLU_updateOutput(
            self._backend.library_state,
            input,
            output,
            self.noise,
            self.lower,
            self.upper,
            self.train,
            self.inplace,
            torch.default_generator if not input.is_cuda else 0
        )
        self.save_for_backward(input)
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = input.new()
        self._backend.RReLU_updateGradInput(
            self._backend.library_state,
            input,
            grad_output,
            grad_input,
            self.noise,
            self.lower,
            self.upper,
            self.train,
            False
        )
        return grad_input


class SELU(InplaceFunction):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    @staticmethod
    def forward(ctx, input, inplace):
        backend = type2backend[type(input)]
        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.new(input.size())
        backend.ELU_updateOutput(
            backend.library_state,
            input,
            output,
            SELU.alpha,
            inplace,
        )
        output.mul_(SELU.scale)
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_variables
        if grad_output.volatile:
            grad_input = Variable(input.data.new(input.size()), volatile=True)
            backend = type2backend[type(input.data)]
            backend.ELU_updateGradInput(
                backend.library_state,
                input.data,
                grad_output.data.mul(SELU.scale),
                grad_input.data,
                output.data.div(SELU.scale),
                SELU.alpha,
                False
            )
        else:
            positive_mask = (output > 0).type_as(grad_output)
            negative_mask = (output <= 0).type_as(grad_output)
            grad_input = grad_output * SELU.scale * (positive_mask +
                                                     negative_mask * (output / SELU.scale + SELU.alpha))
        return grad_input, None


class Softmin(Function):

    def forward(self, input):
        self._backend = type2backend[type(input)]
        self.mininput = input.clone().mul(-1)
        output = input.new()
        self._backend.SoftMax_updateOutput(
            self._backend.library_state,
            self.mininput,
            output
        )
        self.save_for_backward(output)
        return output

    def backward(self, grad_output):
        output, = self.saved_tensors
        grad_input = grad_output.new()
        self._backend.SoftMax_updateGradInput(
            self._backend.library_state,
            self.mininput,
            grad_output,
            grad_input,
            output
        )
        return grad_input.mul(-1)


_all_functions.append(PReLU)
_all_functions.append(PReLUBackward)
_all_functions.append(BatchNorm2)
_all_functions.append(BatchNorm2Backward)
_all_functions.append(BatchNorm2BackwardNotAffine)
_all_functions.append(RReLU)
_all_functions.append(SELU)
_all_functions.append(Softmin)
