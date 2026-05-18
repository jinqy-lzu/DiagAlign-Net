import torch


def get_params(model, ignore_auxiliary_head=True):
    if not ignore_auxiliary_head:
        params = sum([m.numel() for m in model.parameters()])
    else:
        params = sum([m.numel() for k, m in model.named_parameters() if 'auxiliary_head' not in k])
    return params

# def get_flops(model, input_shape=(3, 224, 224)):
#     if hasattr(model, 'flops'):
#         return model.flops(input_shape)
#     else:
#         return get_flops_hook(model, input_shape)

# def get_flops_hook(model, input_shape=(3, 224, 224)):
#     is_training = model.training
#     list_conv = []

#     def conv_hook(self, input, output):
#         batch_size, input_channels, input_height, input_width = input[0].size()
#         output_channels, output_height, output_width = output[0].size()

#         assert self.in_channels % self.groups == 0

#         kernel_ops = self.kernel_size[0] * self.kernel_size[
#             1] * (self.in_channels // self.groups)
#         params = output_channels * kernel_ops
#         flops = batch_size * params * output_height * output_width

#         list_conv.append(flops)

#     list_linear = []

#     def linear_hook(self, input, output):
#         batch_size = input[0].size(0) if input[0].dim() == 2 else 1

#         weight_ops = self.weight.nelement()

#         flops = batch_size * weight_ops
#         list_linear.append(flops)

#     def foo(net, hook_handle):
#         childrens = list(net.children())
#         if not childrens:
#             if isinstance(net, torch.nn.Conv2d):
#                 hook_handle.append(net.register_forward_hook(conv_hook))
#             if isinstance(net, torch.nn.Linear):
#                 hook_handle.append(net.register_forward_hook(linear_hook))
#             return
#         for c in childrens:
#             foo(c, hook_handle)

#     hook_handle = []
#     foo(model, hook_handle)
#     input = torch.rand(*input_shape).unsqueeze(0).to(next(model.parameters()).device)
#     model.eval()
#     with torch.no_grad():
#         out = model(input)
#     for handle in hook_handle:
#         handle.remove()

#     total_flops = sum(sum(i) for i in [list_conv, list_linear])
#     model.train(is_training)
#     return total_flops

def get_flops(model, input_shape):
    """
    计算模型的FLOPs，支持单输入或多输入模型。

    参数：
        model: nn.Module
        input_shapes: tuple，包含一个或多个输入的shape
                      例如：((3, 224, 224),) 或 ((3, 224, 224), (1, 16, 16))
    返回：
        total_flops: 总FLOPs
    """
    if hasattr(model, 'flops'):
        return model.flops(input_shape)
    else:
        return get_flops_hook(model, input_shape)


def get_flops_hook(model, input_shapes):
    """
    使用 forward hook 计算模型的FLOPs（支持多输入）。
    输入：
        input_shapes: tuple，如 ((3, 224, 224),) 或 ((3, 224, 224), (1, 16, 16))
    """
    is_training = model.training
    list_conv, list_linear = [], []

    # hook: Conv2d
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels // self.groups)
        params = output_channels * kernel_ops
        flops = batch_size * params * output_height * output_width
        list_conv.append(flops)

    # hook: Linear
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops = self.weight.nelement()
        flops = batch_size * weight_ops
        list_linear.append(flops)

    # 递归注册hook
    def register_hooks(net, hook_handles):
        children = list(net.children())
        if not children:
            if isinstance(net, torch.nn.Conv2d):
                hook_handles.append(net.register_forward_hook(conv_hook))
            elif isinstance(net, torch.nn.Linear):
                hook_handles.append(net.register_forward_hook(linear_hook))
            return
        for c in children:
            register_hooks(c, hook_handles)

    hook_handles = []
    register_hooks(model, hook_handles)

    # ✅ 根据输入shape创建随机张量
    if not isinstance(input_shapes, (tuple, list)):
        raise TypeError("input_shapes 必须是 tuple 或 list 类型，例如 ((3, 224, 224),)")

    def make_tensor_from_shape(shape):
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"输入必须为 tuple 或 list，而不是 {type(shape)}")
        return torch.rand(1, *shape)

    device = next(model.parameters()).device
    model_inputs = tuple(make_tensor_from_shape(s).to(device) for s in input_shapes)

    # 前向传播统计FLOPs
    model.eval()
    with torch.no_grad():
        _ = model(*model_inputs)

    # 移除hook
    for h in hook_handles:
        h.remove()

    total_flops = sum(sum(i) for i in [list_conv, list_linear])
    model.train(is_training)
    return total_flops
