import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False
        )
        self.pointwise = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Single2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Single2DConv, self).__init__()
        self.conv_1 = DepthwiseSeparableConv(in_dim, out_dim)

    def forward(self, x):
        return self.conv_1(x)


class Double2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Double2DConv, self).__init__()
        self.conv_1 = DepthwiseSeparableConv(in_dim, out_dim)
        self.conv_2 = DepthwiseSeparableConv(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class TrackNetV3NanoOptimized(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TrackNetV3NanoOptimized, self).__init__()
        self.down_block_1 = Single2DConv(in_dim, 8)  # 16 -> 8
        self.down_block_2 = Single2DConv(8, 16)  # 32 -> 16
        self.down_block_3 = Single2DConv(16, 32)  # Double -> Single, 64 -> 32
        self.bottleneck = Single2DConv(32, 64)  # 128 -> 64
        self.up_block_1 = Single2DConv(96, 32)  # 192 -> 96 (32+64), Double -> Single
        self.up_block_2 = Single2DConv(48, 16)  # 96 -> 48 (16+32)
        self.up_block_3 = Single2DConv(24, 8)  # 48 -> 24 (8+16)
        self.predictor = nn.Conv2d(8, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down_block_1(x)  # (N,   8,  288,   512)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)  # (N,   8,  144,   256)
        x2 = self.down_block_2(x)  # (N,  16,  144,   256)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)  # (N,  16,   72,   128)
        x3 = self.down_block_3(x)  # (N,  32,   72,   128)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)  # (N,  32,   36,    64)
        x = self.bottleneck(x)  # (N,  64,   36,    64)
        x = torch.cat(
            [
                F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False),
                x3,
            ],
            dim=1,
        )  # (N,  96,   72,   128)
        x = self.up_block_1(x)  # (N,  32,   72,   128)
        x = torch.cat(
            [
                F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False),
                x2,
            ],
            dim=1,
        )  # (N,  48,  144,   256)
        x = self.up_block_2(x)  # (N,  16,  144,   256)
        x = torch.cat(
            [
                F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False),
                x1,
            ],
            dim=1,
        )  # (N,  24,  288,   512)
        x = self.up_block_3(x)  # (N,   8,  288,   512)
        x = self.predictor(x)  # (N,   3,  288,   512)
        x = self.sigmoid(x)  # (N,   3,  288,   512)
        return x
