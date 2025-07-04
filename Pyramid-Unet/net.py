import torch
from torch import nn
from torch.nn import functional as F


# Define the feature fusion module: PyramidModule
class PyramidModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PyramidModule, self).__init__()

        # Use different convolution kernel sizes to extract multi-scale features
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, padding=0)  # 1x1 convolution
        self.conv2 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)  # 3x3 convolution
        self.bn2 = nn.BatchNorm2d(out_channels // 4)  # Batch Normalization
        self.dropout2 = nn.Dropout2d(0.3)  # Dropout to prevent overfitting
        self.relu2 = nn.LeakyReLU()  # Activation function

        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, stride=1, padding=2)  # 5x5 convolution
        self.bn3 = nn.BatchNorm2d(out_channels // 4)
        self.dropout3 = nn.Dropout2d(0.3)
        self.relu3 = nn.LeakyReLU()

        # Adaptive pooling to reduce feature map size to 1x1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv4 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1,
                               padding=0)  # Global feature extraction

    def forward(self, x):
        # Process features through different convolution kernels
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out2 = self.bn2(out2)
        out2 = self.relu2(out2)
        out3 = self.conv3(x)
        out3 = self.bn3(out3)
        out3 = self.relu3(out3)
        # Use global pooling and then upsample to the original size
        out4 = self.conv4(self.pool(x))
        out4 = F.interpolate(out4, size=out1.size()[2:], mode='bilinear', align_corners=True)

        # Concatenate features extracted from different convolution kernels
        out = torch.cat([out1, out2, out3, out4], dim=1)

        return out


# Define the dual convolution module: Conv_Block
class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1, padding_mode='reflect', bias=False),
            # 3x3 convolution
            nn.BatchNorm2d(out_channel),  # Batch Normalization
            nn.Dropout2d(0.3),  # Dropout to prevent overfitting
            nn.LeakyReLU(),  # Activation function
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),  # Second 3x3 convolution
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


# Define the downsampling module: DownSample
class DownSample(nn.Module):
    def __init__(self, inchannel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, 3, 2, 1, padding_mode='reflect', bias=False),
            # 3x3 convolution with stride=2 for downsampling
            nn.BatchNorm2d(inchannel),
            nn.LeakyReLU()
        )

    # Forward propagation for downsampling
    def forward(self, x):
        return self.layer(x)


# Define the upsampling module: UpSample
class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)  # 1x1 convolution to reduce channel size by half

    def forward(self, x, feature_map):
        # Upsampling using nearest neighbor interpolation
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        # Concatenate the upsampled feature map with the corresponding encoder feature map (skip connection in U-Net)
        return torch.cat((out, feature_map), dim=1)


# Define the U-Net network structure
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder (convolution + downsampling)
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.p1 = PyramidModule(64, 64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.p2 = PyramidModule(128, 128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.p3 = PyramidModule(256, 256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.p4 = PyramidModule(512, 512)
        self.c5 = Conv_Block(512, 1024)
        self.p5 = PyramidModule(1024, 1024)

        # Decoder (convolution + upsampling)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.p6 = PyramidModule(512, 512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.p7 = PyramidModule(256, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.p8 = PyramidModule(128, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.p9 = PyramidModule(64, 64)

        # Output layer, 3-channel output
        self.out = nn.Conv2d(64, 3, 3, 1, 1)
        self.Th = nn.Sigmoid()  # Sigmoid function for normalization

    # Define forward propagation
    def forward(self, x):
        R1 = self.p1(self.c1(x))  # First level of encoder
        R2 = self.p2(self.c2(self.d1(R1)))  # Second level of encoder
        R3 = self.p3(self.c3(self.d2(R2)))  # Third level of encoder
        R4 = self.p4(self.c4(self.d3(R3)))  # Fourth level of encoder
        R5 = self.p5(self.c5(self.d4(R4)))  # Fifth level of encoder

        # Decode and combine features from the encoder (skip connections)
        O1 = self.p6(self.c6(self.u1(R5, R4)))  # First level of decoder
        O2 = self.p7(self.c7(self.u2(O1, R3)))  # Second level of decoder
        O3 = self.p8(self.c8(self.u3(O2, R2)))  # Third level of decoder
        O4 = self.p9(self.c9(self.u4(O3, R1)))  # Fourth level of decoder

        # Output the final prediction
        return self.Th(self.out(O4))


if __name__ == '__main__':
    # Test the network with a random input tensor and check the output shape
    x = torch.randn(2, 3, 256, 256)
    net = UNet()
    print(net(x).shape)  # Print the shape of the output tensor
