import torch
import torch.nn as nn

# class Conv2DBlock(nn.Module):
#     """ Conv2D + BN + ReLU """
#     def __init__(self, in_dim, out_dim, **kwargs):
#         super(Conv2DBlock, self).__init__(**kwargs)
#         self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding='same', bias=False)
#         self.bn = nn.BatchNorm2d(out_dim)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x


class Conv2DBlock(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Conv2DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv2d(
            in_dim, out_dim, kernel_size=3, padding=1, bias=False
        )  # padding=1 for 3x3 kernel
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Double2DConv(nn.Module):
    """ Conv2DBlock x 2 """
    def __init__(self, in_dim, out_dim):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class Triple2DConv(nn.Module):
    """ Conv2DBlock x 3 """
    def __init__(self, in_dim, out_dim):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x

class Single2DConv(nn.Module):
    """ Conv2DBlock x 1 (for Nano) """
    def __init__(self, in_dim, out_dim):
        super(Single2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        return x

class TrackNetV3Small(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TrackNetV3Small, self).__init__()
        self.down_block_1 = Double2DConv(in_dim, 32)  # Уменьшено с 64 до 32
        self.down_block_2 = Double2DConv(32, 64)      # Уменьшено с 128 до 64
        self.down_block_3 = Double2DConv(64, 128)     # Triple2DConv -> Double2DConv, 256 -> 128
        self.bottleneck = Double2DConv(128, 256)      # Triple2DConv -> Double2DConv, 512 -> 256
        self.up_block_1 = Double2DConv(384, 128)      # 768 -> 384 (128+256), Triple -> Double
        self.up_block_2 = Double2DConv(192, 64)       # 384 -> 192 (64+128)
        self.up_block_3 = Double2DConv(96, 32)        # 192 -> 96 (32+64), 64 -> 32
        self.predictor = nn.Conv2d(32, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down_block_1(x)                                       # (N,   32,  288,   512)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)                     # (N,   32,  144,   256)
        x2 = self.down_block_2(x)                                       # (N,   64,  144,   256)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)                     # (N,   64,   72,   128)
        x3 = self.down_block_3(x)                                       # (N,  128,   72,   128)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)                     # (N,  128,   36,    64)
        x = self.bottleneck(x)                                          # (N,  256,   36,    64)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)      # (N,  384,   72,   128)
        x = self.up_block_1(x)                                          # (N,  128,   72,   128)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)      # (N,  192,  144,   256)
        x = self.up_block_2(x)                                          # (N,   64,  144,   256)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)      # (N,   96,  288,   512)
        x = self.up_block_3(x)                                          # (N,   32,  288,   512)
        x = self.predictor(x)                                           # (N,    3,  288,   512)
        x = self.sigmoid(x)                                             # (N,    3,  288,   512)
        return x

class TrackNetV3Nano(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TrackNetV3Nano, self).__init__()
        self.down_block_1 = Single2DConv(in_dim, 16)  # Double -> Single, 64 -> 16
        self.down_block_2 = Single2DConv(16, 32)     # Double -> Single, 128 -> 32
        self.down_block_3 = Double2DConv(32, 64)      # Triple -> Double, 256 -> 64
        self.bottleneck = Single2DConv(64, 128)       # Triple -> Single, 512 -> 128
        self.up_block_1 = Double2DConv(192, 64)       # 768 -> 192 (64+128), Triple -> Double
        self.up_block_2 = Single2DConv(96, 32)        # 384 -> 96 (32+64), Double -> Single
        self.up_block_3 = Single2DConv(48, 16)        # 192 -> 48 (16+32), Double -> Single
        self.predictor = nn.Conv2d(16, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down_block_1(x)                                       # (N,   16,  288,   512)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)                     # (N,   16,  144,   256)
        x2 = self.down_block_2(x)                                       # (N,   32,  144,   256)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)                     # (N,   32,   72,   128)
        x3 = self.down_block_3(x)                                       # (N,   64,   72,   128)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)                     # (N,   64,   36,    64)
        x = self.bottleneck(x)                                          # (N,  128,   36,    64)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)      # (N,  192,   72,   128)
        x = self.up_block_1(x)                                          # (N,   64,   72,   128)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)      # (N,   96,  144,   256)
        x = self.up_block_2(x)                                          # (N,   32,  144,   256)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)      # (N,   48,  288,   512)
        x = self.up_block_3(x)                                          # (N,   16,  288,   512)
        x = self.predictor(x)                                           # (N,    3,  288,   512)
        x = self.sigmoid(x)                                             # (N,    3,  288,   512)
        return x
