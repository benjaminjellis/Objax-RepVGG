"""
RepVGG Models
"""
import objax
from objax import nn
import objax.functional as F
from objax import TrainVar
from objax.constants import ConvPadding
from objax.typing import JaxArray

import jax
import jax.numpy as jn

from typing import Union, Optional, Tuple, Sequence

__all__ = ["create_RepVGG_A0",
           "create_RepVGG_A1",
           "create_RepVGG_A2",
           "create_RepVGG_B0",
           "create_RepVGG_B1",
           "create_RepVGG_B1g2",
           "create_RepVGG_B1g4",
           "create_RepVGG_B2",
           "create_RepVGG_B2g4",
           "create_RepVGG_B3",
           "create_RepVGG_B3g2",
           "create_RepVGG_B3g4",
           "get_RepVGG_func_by_name"
           ]


def conv_bn(in_channels: int,
            out_channels: int,
            k: Union[Tuple[int, int], int],
            strides: Union[Tuple[int, int], int],
            padding: Union[ConvPadding, str, Sequence[Tuple[int, int]], Tuple[int, int], int],
            groups: Optional[int] = 1):
    """
    Create conv bn module
    """
    result = nn.Sequential([
        nn.Conv2D(nin = in_channels,
                  nout = out_channels,
                  k = k,
                  strides = strides,
                  padding = padding,
                  groups = groups,
                  use_bias = False),
        nn.BatchNorm2D(nin = out_channels)
    ])
    return result


class RepVGGBlock(objax.Module):
    """
    Create a RepVGG Block
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[Tuple[int, int], int],
                 stride: Union[Tuple[int, int], int] = 1,
                 padding: Union[ConvPadding, str, Sequence[Tuple[int, int]], Tuple[int, int], int] = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 deploy: bool = False,
                 ):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.get_equivalent_kernel_bias_jit = jax.jit(self.get_equivalent_kernel_bias)

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = F.relu

        if deploy:
            self.rbr_reparam = nn.Conv2D(nin = in_channels,
                                         nout = out_channels,
                                         k = kernel_size,
                                         strides = stride,
                                         padding = padding,
                                         dilations = dilation,
                                         groups = groups,
                                         use_bias = True
                                         )
        else:
            self.rbr_identity = nn.BatchNorm2D(
                nin = in_channels) if out_channels == in_channels and stride == 1 else None

            self.rbr_dense = conv_bn(in_channels = in_channels,
                                     out_channels = out_channels,
                                     k = kernel_size,
                                     strides = stride,
                                     padding = padding,
                                     groups = groups)

            self.rbr_1x1 = conv_bn(in_channels = in_channels,
                                   out_channels = out_channels,
                                   k = 1,
                                   strides = stride,
                                   padding = padding_11,
                                   groups = groups)
            # print('RepVGG Block, identity = ', self.rbr_identity)

    def __call__(self, x: JaxArray, training: bool):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(x))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x, training = training)
        return self.nonlinearity(self.rbr_dense(x, training = training) + self.rbr_1x1(x, training = training) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            npad = ((0, 0), (0, 0), (0, 2), (0, 2))
            return objax.functional.pad(kernel1x1, npad)

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.vars()["(Sequential)[0](Conv2D).w"]
            running_mean = branch.vars()["(Sequential)[1](BatchNorm2D).running_mean"]
            running_var = branch.vars()["(Sequential)[1](BatchNorm2D).running_var"]
            gamma = branch.vars()["(Sequential)[1](BatchNorm2D).gamma"]
            beta = branch.vars()["(Sequential)[1](BatchNorm2D).beta"]
            eps = 1e-06
            kernel = kernel.transpose([3, 2, 1, 0])
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = jn.zeros((self.in_channels, input_dim, 3, 3), dtype = jn.float32)
                for i in range(self.in_channels):
                    kernel_value = jax.ops.index_update(kernel_value, (i, i % input_dim, 1, 1), 1)
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.vars()["(BatchNorm2D).running_mean"]
            running_var = branch.vars()["(BatchNorm2D).running_var"]
            gamma = branch.vars()["(BatchNorm2D).gamma"]
            beta = branch.vars()["(BatchNorm2D).beta"]
            eps = 1e-6
        std = jn.sqrt((running_var + eps))
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        # kernel, bias = self.get_equivalent_kernel_bias_jit()
        kernel, bias = self.get_equivalent_kernel_bias()
        kernel = kernel.transpose([3, 2, 1, 0])
        """
        print((self.rbr_dense.vars()["(Sequential)[0](Conv2D).w"].shape[2] * self.rbr_dense[0].groups,
              self.rbr_dense.vars()["(Sequential)[0](Conv2D).w"].shape[3],
              self.rbr_dense.vars()["(Sequential)[0](Conv2D).w"].shape[0],
              self.rbr_dense[0].strides[0],
              self.rbr_dense[0].padding[0][0],
              self.rbr_dense[0].dilations[0],
               self.rbr_dense[0].groups,
              True))
        """

        self.rbr_reparam = nn.Conv2D(
            nin = self.rbr_dense.vars()["(Sequential)[0](Conv2D).w"].shape[2] * self.rbr_dense[0].groups,
            nout = self.rbr_dense.vars()["(Sequential)[0](Conv2D).w"].shape[3],
            k = self.rbr_dense.vars()["(Sequential)[0](Conv2D).w"].shape[0],
            strides = self.rbr_dense[0].strides[0],
            padding = self.rbr_dense[0].padding[0][0],
            dilations = self.rbr_dense[0].dilations[0],
            groups = self.rbr_dense[0].groups,
            use_bias = True)

        self.rbr_reparam.w = TrainVar(kernel)
        self.rbr_reparam.b = TrainVar(bias.squeeze(axis = 0))
        for para in self.vars():
            jax.lax.stop_gradient(para)
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')


class RepVGG(objax.Module):
    """
    Create RepVGG Model
    """

    def __init__(self,
                 num_blocks: list,
                 num_classes: int = 1000,
                 width_multiplier: Optional[list] = None,
                 override_groups_map = None,
                 deploy = False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels = 3,
                                  out_channels = self.in_planes,
                                  kernel_size = 3,
                                  stride = 2,
                                  padding = 1,
                                  deploy = self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride = 2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride = 2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride = 2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride = 2)
        self.gap = F.average_pool_2d
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels = self.in_planes, out_channels = planes, kernel_size = 3,
                                      stride = stride, padding = 1, groups = cur_groups, deploy = self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(blocks)

    def get_blocks(self):
        stages = [self.stage0, self.stage1, self.stage2, self.stage3, self.stage4]
        blocks = []
        for stage in stages:
            if isinstance(stage, nn.Sequential):
                for s in stage:
                    blocks.append(s)
            else:
                blocks.append(stage)
        return blocks

    def __call__(self, x: JaxArray, training: bool):
        out = self.stage0(x, training = training)
        out = self.stage1(out, training = training)
        out = self.stage2(out, training = training)
        out = self.stage3(out, training = training)
        out = self.stage4(out, training = training)
        size = out.shape[-1]
        out = self.gap(out, size = size)
        out = objax.functional.flatten(out)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def create_RepVGG_A0(deploy = False):
    return RepVGG(num_blocks = [2, 4, 14, 1],
                  num_classes = 1000,
                  width_multiplier = [0.75, 0.75, 0.75, 2.5],
                  override_groups_map = None,
                  deploy = deploy)


def create_RepVGG_A1(deploy = False):
    return RepVGG(num_blocks = [2, 4, 14, 1],
                  num_classes = 1000,
                  width_multiplier = [1, 1, 1, 2.5],
                  override_groups_map = None,
                  deploy = deploy)


def create_RepVGG_A2(deploy = False):
    return RepVGG(num_blocks = [2, 4, 14, 1],
                  num_classes = 1000,
                  width_multiplier = [1.5, 1.5, 1.5, 2.75],
                  override_groups_map = None,
                  deploy = deploy)


def create_RepVGG_B0(deploy = False):
    return RepVGG(num_blocks = [4, 6, 16, 1],
                  num_classes = 1000,
                  width_multiplier = [1, 1, 1, 2.5],
                  override_groups_map = None,
                  deploy = deploy)


def create_RepVGG_B1(deploy = False):
    return RepVGG(num_blocks = [4, 6, 16, 1],
                  num_classes = 1000,
                  width_multiplier = [2, 2, 2, 4],
                  override_groups_map = None,
                  deploy = deploy)


def create_RepVGG_B1g2(deploy = False):
    return RepVGG(num_blocks = [4, 6, 16, 1],
                  num_classes = 1000,
                  width_multiplier = [2, 2, 2, 4],
                  override_groups_map = g2_map,
                  deploy = deploy)


def create_RepVGG_B1g4(deploy = False):
    return RepVGG(num_blocks = [4, 6, 16, 1],
                  num_classes = 1000,
                  width_multiplier = [2, 2, 2, 4],
                  override_groups_map = g4_map,
                  deploy = deploy)


def create_RepVGG_B2(deploy = False):
    return RepVGG(num_blocks = [4, 6, 16, 1],
                  num_classes = 1000,
                  width_multiplier = [2.5, 2.5, 2.5, 5],
                  override_groups_map = None,
                  deploy = deploy)


def create_RepVGG_B2g2(deploy = False):
    return RepVGG(num_blocks = [4, 6, 16, 1],
                  num_classes = 1000,
                  width_multiplier = [2.5, 2.5, 2.5, 5],
                  override_groups_map = g2_map,
                  deploy = deploy)


def create_RepVGG_B2g4(deploy = False):
    return RepVGG(num_blocks = [4, 6, 16, 1],
                  num_classes = 1000,
                  width_multiplier = [2.5, 2.5, 2.5, 5],
                  override_groups_map = g4_map,
                  deploy = deploy)


def create_RepVGG_B3(deploy = False):
    return RepVGG(num_blocks = [4, 6, 16, 1],
                  num_classes = 1000,
                  width_multiplier = [3, 3, 3, 5],
                  override_groups_map = None,
                  deploy = deploy)


def create_RepVGG_B3g2(deploy = False):
    return RepVGG(num_blocks = [4, 6, 16, 1],
                  num_classes = 1000,
                  width_multiplier = [3, 3, 3, 5],
                  override_groups_map = g2_map,
                  deploy = deploy)


def create_RepVGG_B3g4(deploy = False):
    return RepVGG(num_blocks = [4, 6, 16, 1],
                  num_classes = 1000,
                  width_multiplier = [3, 3, 3, 5],
                  override_groups_map = g4_map,
                  deploy = deploy)


func_dict = {
    'RepVGG-A0': create_RepVGG_A0,
    'RepVGG-A1': create_RepVGG_A1,
    'RepVGG-A2': create_RepVGG_A2,
    'RepVGG-B0': create_RepVGG_B0,
    'RepVGG-B1': create_RepVGG_B1,
    'RepVGG-B1g2': create_RepVGG_B1g2,
    'RepVGG-B1g4': create_RepVGG_B1g4,
    'RepVGG-B2': create_RepVGG_B2,
    'RepVGG-B2g2': create_RepVGG_B2g2,
    'RepVGG-B2g4': create_RepVGG_B2g4,
    'RepVGG-B3': create_RepVGG_B3,
    'RepVGG-B3g2': create_RepVGG_B3g2,
    'RepVGG-B3g4': create_RepVGG_B3g4,
}


def get_RepVGG_func_by_name(name):
    return func_dict[name]
