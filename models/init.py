import math

from torch import nn


def basic_init(m: nn.Module):
    classname = m.__class__.__name__

    if 'Conv' in classname:
        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()


def efficiency_init(m: nn.Module):
    """
    refer to
     - Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, https://arxiv.org/pdf/1706.02677.pdf

    For BN layers, the learnable scaling coefficient γ is initialized to be 1,
    except for each residual block’s last BN
    where γ is initialized to be 0. Setting γ = 0 in the last BN of
    each residual block causes the forward/backward signal initially to propagate
    through the identity shortcut of ResNets,
    which we found to ease optimization at the start of training.
    This initialization improves all models but is particularly
    helpful for large minibatch training as we will show.
    """

    classname = m.__class__.__name__

    if 'Conv' in classname:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
