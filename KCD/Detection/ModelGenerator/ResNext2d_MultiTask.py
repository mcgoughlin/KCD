from typing import Any, Callable, List, Optional, Type, Union,Dict
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Conv2d
from typing import Any, Callable, List, Optional, Type, Union,Dict
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Conv2d
from torchvision.models import resnext101_32x8d,ResNeXt101_32X8D_Weights

# code adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2):
        super().__init__()
        # print("kernel size:",kernel_size)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size, stride=kernel_size,
                                       bias=False)

        # print(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)


    def forward(self, xb):
        # print("UpConv before",xb.shape)
        # print("UpConv after",self.conv(xb).shape)
        return nn.functional.relu(self.conv(xb))

class ResNet2d_MTL(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        num_seg_classes: int = 4,
        in_channels = 1,
        depth_z = 20,
        internal_seg_channels=8,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.upconvs = nn.ModuleList()
        self.upconvs.append(UpConv(64, internal_seg_channels, kernel_size=2))
        self.upconvs.append(UpConv(64, internal_seg_channels, kernel_size=4))
        filters = 256
        for i in range(4):
            self.upconvs.append(UpConv(filters, internal_seg_channels, kernel_size = filters//64))
            filters*=2

        self.seg_conv_pool = nn.AdaptiveMaxPool2d((224,224))
        self.seg_conv_1 = nn.Conv2d(internal_seg_channels*6, 2*internal_seg_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.seg_conv_2 = nn.Conv2d(2*internal_seg_channels, num_seg_classes, kernel_size=1, stride=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x2 = self.maxpool(x1)

        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)
        x_out = self.avgpool(x6)
        x_out = torch.flatten(x_out, 1)
        x_out = self.fc(x_out)

        if self.training:
            ##### Segmentation MTL #####
            seg_outs = []
            for i,out in enumerate([x1,x2,x3,x4,x5,x6]):
                seg_outs.append(self.seg_conv_pool(self.upconvs[i](out)))
            seg_out = torch.cat(seg_outs,1)
            # interpolate mid and high to match shape of base
            seg_out = self.seg_conv_2(nn.functional.relu(self.seg_conv_1(seg_out)))
            return x_out, seg_out
        else:
            return x_out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _resnet2d_mtl(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    in_channels: int = 1,
    num_classes: int = 1000,
    **kwargs: Any,
) -> ResNet2d_MTL:


    model = ResNet2d_MTL(block, layers,in_channels=in_channels,num_classes=num_classes, **kwargs)

    return model
def resnext502d_32x4d(in_channels=1,num_classes=3,**kwargs) -> ResNet2d_MTL:
    # _ovewrite_named_param(kwargs, "groups", 32)
    groups = 32
    # _ovewrite_named_param(kwargs, "width_per_group", 4)
    width_per_group = 4
    return _resnet2d_mtl(Bottleneck, [3, 4, 6, 3],in_channels=in_channels,num_classes=num_classes,
                     groups = groups, width_per_group=width_per_group,**kwargs)

def resnext1012d_32x8d(in_channels=1,num_classes=3,**kwargs) -> ResNet2d_MTL:
    groups = 32
    width_per_group = 8

    model = resnext101_32x8d(ResNeXt101_32X8D_Weights)

    mtlmodel = _resnet2d_mtl(Bottleneck, [3, 4, 23, 3], in_channels=in_channels, num_classes=num_classes,
                             groups=groups, width_per_group=width_per_group, **kwargs)

    # load pretrained weights into mtl model
    model_dict = model.state_dict()
    mtlmodel_dict = mtlmodel.state_dict()
    # 1. filter out unnecessary keys and keys that only exist in mtlmodel_dict
    pretrained_dict = {k: v for k, v in model_dict.items() if k in mtlmodel_dict}
    # 1.5. filter out final fc layer and first conv layer
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if not (k.startswith('fc') or k.startswith('conv1'))}
    # 2. overwrite entries in the existing state dict
    mtlmodel_dict.update(pretrained_dict)
    # 3. load the new state dict
    mtlmodel.load_state_dict(mtlmodel_dict)

    return mtlmodel


# test on dummy image
if __name__ == "__main__":
    model = resnext1012d_32x8d(in_channels=1,num_classes=3,num_seg_classes=4)
    model.train()

    from KCD.Detection.Dataloaders.sliceseg_dataloader import SW_Data_seglabelled
    dataset = SW_Data_seglabelled(path="/Users/mcgoug01/Downloads/Data",name="kits23_nooverlap",device='cpu',depth_z=1,boundary_z=1)
    dataset.apply_foldsplit()

    im,(seg,label) = dataset[0]
    im = im.unsqueeze(0)
    pred_lb,pred_seg = model(im)

    seg_loss_func = nn.CrossEntropyLoss()
    pred_loss_func = nn.CrossEntropyLoss()

    pred_loss = pred_loss_func(pred_lb,label)
    seg_loss = seg_loss_func(pred_seg,seg.unsqueeze(0))

    print(pred_loss,seg_loss)