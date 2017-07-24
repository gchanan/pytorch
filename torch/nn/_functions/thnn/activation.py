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
            #gI =

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
        print("ctx.affine is", ctx.affine)
        if ctx.affine:
            input, gamma, beta = ctx.saved_variables
        else:
            input, = ctx.saved_variables
            gamma = beta = None

        print("trying to apply", input, gamma, beta, grad_output, ctx.eps)
        if ctx.affine:
            ret = BatchNorm2Backward.apply(input, gamma, beta, grad_output, ctx.eps)
        else:
            print("in ret")
            ret = BatchNorm2BackwardNotAffine.apply(input, grad_output, ctx.eps)
            ret = (ret,) + (None, None, None)
            print("done ret")
        print("ret is", ret)
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


#def backback_affine(input, ggI, ggG, ggB, ggO, eps):
    

def back_not_affine(input, grad_output, eps):
    M = input.size(0)
    mu = input.sum(dim=0).div(M).expand_as(input)
    sigma2 = (input - mu).pow(2).sum(dim=0).div(M)


    first_half = (1 / (sigma2 + eps).sqrt()).div(M).expand_as(grad_output)
    second_half = M * grad_output - grad_output.sum(dim=0).expand_as(grad_output) - (input - mu).div((sigma2 + eps).expand_as(grad_output)) * (grad_output * (input - mu)).sum(dim =0).expand_as(grad_output)
    grad_input = first_half * second_half

    grad_beta = None
    grad_gamma = None

    return grad_input, grad_gamma, grad_beta

def indicator(a, b):
    return 1 if a==b else 0

def backback_not_affine(input, gamma, ggI, ggG, ggB, gO, eps):
    M = input.size(0)
    mu = input.sum(dim=0).div(M)
    sigma2 = (input - mu).pow(2).sum(dim=0).div(M)
    affine = gamma is not None

    if affine:
        ggG_expanded = ggG
        while len(ggG_expanded.size()) < len(input.size()) - 1:
            ggG_expanded = ggG_expanded.unsqueeze(1)
        ggB_expanded = ggB
        while len(ggB_expanded.size()) < len(input.size()) - 1:
            ggB_expanded = ggB_expanded.unsqueeze(1)

    if not affine:
        gamma_expanded = 1
    else:
        gamma_expanded = gamma
        while len(gamma_expanded.size()) < len(input.size()) - 1:
            gamma_expanded = gamma_expanded.unsqueeze(1)


    first_half = (gamma_expanded / (sigma2 + eps).sqrt()).div(M).expand_as(ggI)
    grad_second_half = M * ggI - ggI.sum(dim=0).expand_as(ggI) - (input - mu).div((sigma2 + eps).expand_as(ggI)) * (ggI * (input - mu)).sum(dim =0).expand_as(ggI)
    ggO = first_half * grad_second_half
    
    if affine:
        #something = (input-mu).sum(dim=0) * (sigma2 + eps).pow(-1/2)
        #ggO_G_term = ggG_expanded * something
        #ggO_B_term = ggB_expanded.sum(dim=0)
        #while len(ggO_B_term.size()) > 1:
        #    ggO_B_term =  ggO_B_term.sum(dim=1)
        ggO_G_term = ggG_expanded * (input - mu) * (sigma2 + eps).pow(-1/2)
        ggO_B_term = ggB_expanded
        
        ggO = ggO + ggO_G_term + ggO_B_term

    sig2_neg_3_2 = (sigma2 + eps).pow(-3 / 2)
    insig32 = (input - mu) * sig2_neg_3_2
    gOinmu_sum = (gO * (input - mu)).sum(dim=0)
    ggIinmu_sum = (ggI * (input - mu)).sum(dim=0)

    grad_terms = (1 / M * ggI.sum(dim=0) * gO.sum(dim=0) - (gO * ggI).sum(dim=0) +
        3 / M * (sigma2 + eps).pow(-1) * gOinmu_sum * ggIinmu_sum)
    combined_term = 1 / M * insig32 * (grad_terms)
    ggI_sum_terms = 1 / M * ggIinmu_sum * sig2_neg_3_2 * (1 / M * gO.sum(dim=0) - gO)
    ggO_sum_terms = 1 / M * gOinmu_sum * sig2_neg_3_2 * (1 / M * ggI.sum(dim=0) - ggI)

    gI = combined_term + ggI_sum_terms + ggO_sum_terms
    gI = gamma_expanded * gI

    if affine:
        gI_gamma = 0
        first_term = gO * (sigma2 + eps).pow(-1 / 2)
        second_term = - 1 / M * (sigma2 + eps).pow(-1/2) * (gO).sum(dim=0)
        third_term = -1 / M * (input - mu) * (sigma2 + eps).pow(-3 / 2) * (gO * (input - mu)).sum(dim=0)
        ggG_expanded = ggG
        while len(ggG_expanded.size()) < len(input.size()) - 1:
            ggG_expanded = ggG_expanded.unsqueeze(1)
        ggB_expanded = ggB
        while len(ggB_expanded.size()) < len(input.size()) - 1:
            ggB_expanded = ggB_expanded.unsqueeze(1)
        print("sizes", first_term.size(), second_term.size(), third_term.size())
        gI_gamma = ggG_expanded * (first_term + second_term + third_term)
        gI = gI + gI_gamma



    gG = None
    if affine:
        my_first_half = (1 / (sigma2 + eps).sqrt()).div(M).expand_as(gO)
        my_second_half = M * gO - gO.sum(dim=0).expand_as(gO) - (input - mu).div((sigma2 + eps).expand_as(gO)) * (gO * (input - mu)).sum(dim =0).expand_as(gO)
        gG = ggI * my_first_half * my_second_half
        # sum gG over all non-1 dimensions
        #print("before sum", gG.size(), ggG.size())
        gG = gG.sum(dim=0)
        while len(gG.size()) > 1:
            gG =  gG.sum(dim=1)

    #print("returning sizes", gI.size(), gG.size(), ggO.size(), "gamma_size", gamma.size())
    return gI, gG, ggO

class BatchNorm2Backward(Function):
    @staticmethod
    def forward(ctx, input, gamma, beta, grad_output, eps):
        #assert gamma.dim() == 0
        #assert beta.dim() == 0
        #gamma = None
        #beta = None
        print("in batchnorm2backward")
        ctx.save_for_backward(input, gamma, grad_output)
        ctx.eps = eps
        return back(input, gamma, beta, grad_output, eps)
        """ctx.affine = gamma is not None
        M = input.size(0)
        mu = input.sum(dim=0).div(M).expand_as(input)
        sigma2 = (input - mu).pow(2).sum(dim=0).div(M)

        if not ctx.affine:
            gamma_expanded = 1
        else:
            gamma_expanded = gamma
            while len(gamma_expanded.size()) < len(input.size()) - 1:
                gamma_expanded = gamma_expanded.unsqueeze(1)
        
        
        first_half = (gamma_expanded / (sigma2 + eps).sqrt()).div(M).expand_as(grad_output)
        second_half = M * grad_output - grad_output.sum(dim=0).expand_as(grad_output) - (input - mu).div((sigma2 + eps).expand_as(grad_output)) * (grad_output * (input - mu)).sum(dim =0).expand_as(grad_output)
        grad_input = first_half * second_half

        if ctx.affine:
            grad_beta = grad_output.sum(dim=0)
            while len(grad_beta.size()) > 1:
                grad_beta =  grad_beta.sum(dim=1)
        else:
            grad_beta = None
        if ctx.affine:
            grad_gamma = ((input - mu) / (sigma2 + eps).sqrt().expand_as(grad_output) * grad_output).sum(dim = 0)
            while len(grad_gamma.size()) > 1:
                grad_gamma =  grad_gamma.sum(dim=1)
        else:
            grad_gamma = None

        return grad_input, grad_gamma, grad_beta"""

    @staticmethod
    def backward(ctx, ggI, ggG, ggB):
        input, gamma, gO = ctx.saved_variables
        ret = backback_not_affine(input, gamma, ggI, ggG, ggB, gO, ctx.eps)
        return ret[0], ret[1], None, ret[2], None
        #raise ValueError("BacktchNorm2BackwardBackward not implemented yet")
        #assert


class BatchNorm2BackwardNotAffine(Function):
    @staticmethod
    def forward(ctx, input, grad_output, eps):
        print("in batchnorm2backwardnotaffine")
        ret = back_not_affine(input, grad_output, eps)
        ctx.save_for_backward(input, grad_output)
        ctx.eps = eps
        ret2 = ret[0]
        print("ret2", ret2)
        return ret2

    @staticmethod
    def backward(ctx, ggI):
        input, grad_output = ctx.saved_variables
        ret = backback_not_affine(input, None, ggI, None, None, grad_output, ctx.eps)
        return ret[0], ret[2], None
        #raise ValueError("BacktchNorm2BackwardNotAffineBackward not implemented yet")

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
