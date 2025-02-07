import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from .modules import DoubleConv


class SimpleUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[64, 128, 256, 512],
        bott_features=[],
    ):
        super(SimpleUNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.bottleneck = nn.Sequential()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of SimpleUNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of SimpleUNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        bott_in = features[-1]
        for feature in bott_features:
            self.bottleneck.append(DoubleConv(bott_in, feature))
            bott_in = feature

        self.bottleneck.append(DoubleConv(bott_in, features[-1] * 2))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
