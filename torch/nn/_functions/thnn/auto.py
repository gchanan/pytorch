from itertools import repeat
from collections import defaultdict

import torch
from torch._thnn.utils import parse_header, THNN_H_PATH
from torch.autograd import Variable
from torch.autograd.function import Function, InplaceFunction, once_differentiable
from torch._thnn import type2backend

from . import _all_functions


def _make_function_class_criterion(class_name, update_output, update_grad_input, acc_grad_parameters):
    weight_arg_idx = -1
    for i, arg in enumerate(update_output.arguments):
        if arg.name.startswith('weight'):
            weight_arg_idx = i
            break

    buffers_idx = []
    additional_arg_idx = 0
    for arg in update_output.arguments[4:]:
        if not arg.name.startswith('weight') and arg.type == 'THTensor*':
            buffers_idx.append(additional_arg_idx)
        additional_arg_idx += 1

    @staticmethod
    def backward_cls_forward(ctx, input, target, grad_output, additional_args_ctx, backend_ctx):
        ctx.additional_args = additional_args_ctx
        ctx._backend = backend_ctx
        grad_input = grad_output.new().resize_as_(input).zero_()
        getattr(ctx._backend, update_grad_input.name)(ctx._backend.library_state, input, target,
                                                      grad_input, *ctx.additional_args)
        grad_output_expanded = grad_output.view(*repeat(1, grad_input.dim()))
        grad_input.mul_(grad_output_expanded.expand_as(grad_input))
        return grad_input

    @staticmethod
    @once_differentiable
    def backward_cls_backward(ctx, *grad_params):
        raise ValueError(class_name + " can only be differentiated once.")

    backward_cls = type(class_name + "Backward", (Function,),
                        dict(forward=backward_cls_forward, backward=backward_cls_backward))

    @staticmethod
    def forward(ctx, input, target, *args):
        ctx._backend = type2backend[type(input)]
        ctx.save_for_backward(input, target)
        if weight_arg_idx >= 0:
            ctx.weight = args[0]
            args = args[1:]
            ctx.additional_args = list(args)
            insert_idx = weight_arg_idx - 4  # state, input, target, output
            ctx.additional_args.insert(insert_idx, ctx.weight)
        else:
            ctx.additional_args = list(args)

        ctx.forward_args_count = len(ctx.additional_args)
        for idx in buffers_idx:
            ctx.additional_args.insert(idx, input.new(1))
        output = input.new(1)
        getattr(ctx._backend, update_output.name)(ctx._backend.library_state, input, target,
                                                  output, *ctx.additional_args)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_variables
        return ((backward_cls.apply(input, target, grad_output, ctx.additional_args, ctx._backend),) +
                (None,) * (ctx.forward_args_count + 1))

    return type(class_name, (Function,), dict(forward=forward, backward=backward)), backward_cls


def _find_buffers(args, ignored_args):
    additional_arg_idx = 0
    buffers = []
    for arg in args:
        if arg.name in ignored_args:
            continue
        if arg.type == 'THTensor*':
            buffers.append((additional_arg_idx, arg.name))
        additional_arg_idx += 1
    return buffers


def _make_function_class(class_name, update_output, update_grad_input, acc_grad_parameters, double_backwards_function):
    def has_argument(fn, name):
        for arg in fn.arguments:
            if arg.name == name:
                return True
        return False
    save_output = has_argument(update_grad_input, 'output')

    param_args = {'weight', 'bias'}
    ignored_args = {'weight', 'bias', 'gradWeight', 'gradBias', 'output'}
    expected_params = [arg for arg in update_output.arguments[3:]
                       if arg.name in param_args]
    buffers = {}
    buffers['update_output'] = _find_buffers(update_output.arguments[3:],
                                             ignored_args)
    buffers['update_grad_input'] = _find_buffers(
        update_grad_input.arguments[4:], ignored_args)
    if acc_grad_parameters is not None:
        buffers['acc_grad_parameters'] = _find_buffers(
            acc_grad_parameters.arguments[3:], ignored_args)

    # This assumes that only the last argument can be
    # an inplace flag
    is_inplace = update_output.arguments[-1].name == 'inplace'


    @staticmethod
    def _initialize_buffers(ctx, fn_name):
        additional_args = ctx.additional_args
        for idx, name in buffers[fn_name]:
            # TODO: some buffers are necessary only for update output and can be
            # freed right afterwards
            buffer = ctx.buffers[name]
            additional_args = additional_args[:idx] + [buffer] + additional_args[idx:]
        return tuple(additional_args)

    @staticmethod
    def backward_cls_forward(ctx, input, grad_output, additional_args_ctx, backend_ctx, buffers_ctx, *params):
        ctx.additional_args = additional_args_ctx
        ctx.buffers = buffers_ctx
        ctx._backend = backend_ctx
        ctx.save_for_backward(input, grad_output)
        if save_output:
            output = params[0]
            params = params[1:]

        grad_params = tuple(None for p in params)
        grad_input_tuple = (None,)
        if is_inplace:
            ctx.inplace = additional_args_ctx[-1]

        if ctx.needs_input_grad[0]:
            additional_args = _initialize_buffers.__func__(ctx, 'update_grad_input')
            if save_output:
                additional_args = (output,) + additional_args

            if is_inplace and ctx.inplace:
                assert additional_args[-1] is True
                tmp_args = list(additional_args)
                tmp_args[-1] = False
                additional_args = tuple(tmp_args)
            grad_input = input.new(input.size())
            params_without_bias = params if len(params) < 2 else params[:1]
            update_grad_input_fn = getattr(ctx._backend, update_grad_input.name)
            gi_args = params_without_bias + additional_args
            update_grad_input_fn(ctx._backend.library_state, input, grad_output, grad_input, *gi_args)
            grad_input_tuple = (grad_input,)

        if acc_grad_parameters and any(ctx.needs_input_grad[1:]):
            additional_args = _initialize_buffers.__func__(ctx, 'acc_grad_parameters')
            grad_params = tuple(p.new(p.size()).zero_() for p in params)
            appended_grads = len(expected_params) - len(grad_params)
            grad_params += (None,) * appended_grads
            acc_grad_parameters_fn = getattr(ctx._backend, acc_grad_parameters.name)
            param_args = grad_params + additional_args + (1,)
            acc_grad_parameters_fn(ctx._backend.library_state, input, grad_output, *param_args)
            if appended_grads:
                grad_params = grad_params[:-appended_grads]

        return grad_input_tuple + grad_params

    @staticmethod
    def backward_cls_backward(ctx, *grad_params):
        if double_backwards_function is None:
            raise ValueError(class_name + " can only be differentiated once.")
        else:
            input, grad_output = ctx.saved_variables
            #print("comparing grad outputs", grad_output, *grad_params)
            return double_backwards_function(input, *grad_params, *ctx.additional_args), None, None, None, None
            #raise ValueError("would use function here")
            
            #double_backwards_function()

    base_class = Function if not is_inplace else InplaceFunction
    backward_cls = type(class_name + "Backward", (base_class,), dict(forward=backward_cls_forward,
                                                                     backward=backward_cls_backward,
                                                                     _initialize_buffers=_initialize_buffers))

    @staticmethod
    def forward(ctx, input, *params):
        ctx._backend = type2backend[type(input)]

        ctx.additional_args = []
        tensor_params = []
        for param in params:
            if torch.is_tensor(param):
                if type(param) != type(input):
                    raise RuntimeError("input type ({}) doesn't match the type of "
                                       "a parameter tensor ({})".format(torch.typename(input),
                                                                        torch.typename(param)))
                tensor_params.append(param)
            else:
                ctx.additional_args.append(param)

        if is_inplace:
            ctx.inplace = params[-1]
        # Allocate temporary buffers and insert them into additional_args
        ctx.buffers = defaultdict(type(input))
        additional_args = _initialize_buffers.__func__(ctx, 'update_output')

        # Fill in optional params with None
        args = tensor_params[:]
        for i in range(len(params), len(expected_params)):
            param = expected_params[i]
            if param.is_optional:
                args += (None,)
            else:
                raise ValueError("missing required argument '%s'" % param.name)

        args += tuple(additional_args)

        # If the module is working in-place it's output will be set to the
        # same storage as input, but it's variable won't be dirty.
        if is_inplace and ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.new()

        if save_output:
            ctx.save_for_backward(input, output, *tensor_params)
        else:
            ctx.save_for_backward(input, *tensor_params)

        if not ctx.requires_grad:
            del ctx.buffers

        getattr(ctx._backend, update_output.name)(ctx._backend.library_state, input, output, *args)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        t = ctx.saved_variables
        input, params = t[0], t[1:]
        # it may be simpler to recalculate some of these parameters in the Backward function's forward?
        return (backward_cls.apply(input, grad_output, ctx.additional_args, ctx._backend, ctx.buffers, *params) +
                (None,) * len(ctx.needs_input_grad[1:]))

    @staticmethod
    def backward_hardtanh(ctx, grad_output):
        t = ctx.saved_variables
        input, params = t[0], t[1:]
        max_mask = input <= 1
        min_mask = input <= -1
        ret = grad_output * (max_mask - min_mask).type_as(grad_output)
        return ret, None, None, None

    #if class_name != 'Hardtanh':
    return type(class_name, (base_class,), dict(forward=forward, backward=backward,
                                                _initialize_buffers=_initialize_buffers)), backward_cls
    #else:
    #return type(class_name, (base_class,), dict(forward=forward, backward=backward_hardtanh,
    #                                                _initialize_buffers=_initialize_buffers)), None


def _generate_function_classes(scope_dict):
    global function_list, function_by_name
    function_list = parse_header(THNN_H_PATH)
    function_by_name = {fn.name: fn for fn in function_list}
    classes_to_generate = {fn.name.partition('_')[0] for fn in function_list}
    exceptions = {
        'Linear',
        'IndexLinear',
        'SpatialFullConvolution',
        'SpatialConvolutionMM',
        'SparseLinear',
        'TemporalConvolution',
        'SpatialAveragePooling',
        'SpatialMaxPooling',
        'SpatialDilatedMaxPooling',
        'SpatialMaxUnpooling',
        'SpatialAdaptiveMaxPooling',
        'SpatialAdaptiveAveragePooling',
        'VolumetricAveragePooling',
        'VolumetricMaxPooling',
        'VolumetricMaxUnpooling',
        'VolumetricConvolution',
        'VolumetricFullConvolution',
        'VolumetricConvolutionMM',
        'TemporalMaxPooling',
        'BatchNormalization',
        'LookupTable',
        'LookupTableBag',
        'PReLU',
        'RReLU',
        'Threshold',
        'LeakyReLU',
        'GRUFused',
        'LSTMFused',
        'unfolded',
    }
    name_remap = {
        'TemporalConvolution': 'Conv1d',
        'SpatialDilatedConvolution': 'DilatedConv2d',
        'SpatialMaxUnpooling': 'MaxUnpool2d',
        'SpatialReflectionPadding': 'ReflectionPad2d',
        'SpatialReplicationPadding': 'ReplicationPad2d',
        'VolumetricReplicationPadding': 'ReplicationPad3d',
        'VolumetricMaxUnpooling': 'MaxUnpool3d',
        'SoftMax': 'Softmax',
        'LogSoftMax': 'LogSoftmax',
        'HardTanh': 'Hardtanh',
        'HardShrink': 'Hardshrink',
        'SoftPlus': 'Softplus',
        'SoftShrink': 'Softshrink',
        'MSECriterion': 'MSELoss',
        'AbsCriterion': 'L1Loss',
        'BCECriterion': '_BCELoss',  # TODO: move the glue code into THNN
        'ClassNLLCriterion': 'NLLLoss',
        'DistKLDivCriterion': 'KLDivLoss',
        'SpatialClassNLLCriterion': 'NLLLoss2d',
        'MultiLabelMarginCriterion': 'MultiLabelMarginLoss',
        'MultiMarginCriterion': 'MultiMarginLoss',
        'SmoothL1Criterion': 'SmoothL1Loss',
        'SoftMarginCriterion': 'SoftMarginLoss',
    }
    def hardtanh(input, grad_output, *args, **kwargs):
        print("args to hardtanh", *args)
        max_mask = input <= 1
        min_mask = input <= -1
        print("max_mask", max_mask - min_mask)
        ret = grad_output * (max_mask - min_mask).type_as(grad_output)
        print("return from double backwards", ret)
        return ret

    double_backwards_function = {
        #'ReLU6': lambda x: x,
        'Hardtanh': hardtanh  #lambda x: (Variable(x.data.new(x.size())), None, None, None, None)
    }

    classes_to_generate -= exceptions
    for fn in classes_to_generate:
        update_output = function_by_name[fn + '_updateOutput']
        update_grad_input = function_by_name[fn + '_updateGradInput']
        acc_grad_parameters = function_by_name.get(fn + '_accGradParameters')
        class_name = name_remap.get(fn, fn)
        print("class_name is", class_name)
        # This has to call a function to retain correct references to functions
        is_criterion_fn = 'Criterion' in fn
        if is_criterion_fn:
            cls, backward_cls = _make_function_class_criterion(class_name, update_output,
                                                               update_grad_input, acc_grad_parameters)
        else:
            cls, backward_cls = _make_function_class(class_name, update_output,
                                                     update_grad_input, acc_grad_parameters,
                                                     double_backwards_function.get(class_name))
        scope_dict[class_name] = cls
        if backward_cls != None:
            scope_dict[class_name + 'Backward'] = backward_cls
        if not class_name.startswith('_'):
            _all_functions.append(cls)
            if backward_cls != None:
                _all_functions.append(backward_cls)


_generate_function_classes(locals())
