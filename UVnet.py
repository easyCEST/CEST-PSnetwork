import torch
from torch import nn


class ResBlock(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(ResBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(input_channel, output_channel, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.InstanceNorm3d(output_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(output_channel, output_channel, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.InstanceNorm3d(output_channel),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(input_channel, output_channel, kernel_size=1, stride=1),
            nn.InstanceNorm3d(output_channel),
        )

    def forward(self, input_layer):
        layer1 = self.block1(input_layer) + self.block2(input_layer)
        layer2 = nn.LeakyReLU(inplace=True)(layer1)

        return layer2


class DownBlock(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(DownBlock, self).__init__()
        self.block1 = ResBlock(input_channel, output_channel)
        self.block2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, input_layer):
        layer1 = self.block1(input_layer)
        layer2 = self.block2(layer1)

        return layer1, layer2


class UpBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(UpBlock, self).__init__()
        self.block1 = ResBlock(2 * input_channel, input_channel)
        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True),
            nn.Conv3d(input_channel, output_channel, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        )

    def forward(self, input_layer):
        layer1 = self.block1(input_layer)
        layer2 = self.block2(layer1)
        return layer2


class ResUnet3d(nn.Module):
    def __init__(self, input_channel, output_channel, base_channel):
        super(ResUnet3d, self).__init__()
        self.down1 = DownBlock(input_channel, base_channel)
        self.down2 = DownBlock(base_channel, 2 * base_channel)
        self.down3 = DownBlock(2 * base_channel, 4 * base_channel)
        self.down4 = DownBlock(4 * base_channel, 8 * base_channel)
        self.bottom = nn.Sequential(
            ResBlock(8 * base_channel, 16 * base_channel),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True),
            nn.Conv3d(16 * base_channel, 8 * base_channel, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        )
        self.up4 = UpBlock(8 * base_channel, 4 * base_channel)
        self.up3 = UpBlock(4 * base_channel, 2 * base_channel)
        self.up2 = UpBlock(2 * base_channel, base_channel)
        self.outputs = nn.Sequential(
            ResBlock(2 * base_channel, base_channel),
            nn.Conv3d(base_channel, output_channel, kernel_size=1, stride=1),
            nn.InstanceNorm3d(output_channel),
        )

    def forward(self, input_layer):
        cat1, down1 = self.down1(input_layer)
        cat2, down2 = self.down2(down1)
        cat3, down3 = self.down3(down2)
        cat4, down4 = self.down4(down3)
        bottom = self.bottom(down4)
        up4 = self.up4(torch.cat([cat4, bottom], dim=1))
        up3 = self.up3(torch.cat([cat3, up4], dim=1))
        up2 = self.up2(torch.cat([cat2, up3], dim=1))
        output_layer = self.outputs(torch.cat([cat1, up2], dim=1))

        return output_layer


class UVUnet3d(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_channel, layers=4):
        super(UVUnet3d, self).__init__()
        self.blockV = nn.Sequential(
            nn.Conv2d(input_channel, hidden_channel, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.InstanceNorm2d(hidden_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.InstanceNorm2d(hidden_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_channel, output_channel, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.InstanceNorm2d(output_channel),
        )
        self.down1 = DownBlock(input_channel, hidden_channel)
        self.down2 = DownBlock(hidden_channel, 2 * hidden_channel)
        self.down3 = DownBlock(2 * hidden_channel, 4 * hidden_channel)
        self.bottom = nn.Sequential(
            ResBlock(4 * hidden_channel, 8 * hidden_channel),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True),
            nn.Conv3d(8 * hidden_channel, 4 * hidden_channel, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        )
        self.up3 = UpBlock(4 * hidden_channel, 2 * hidden_channel)
        self.up2 = UpBlock(2 * hidden_channel, hidden_channel)
        self.outputs = nn.Sequential(
            ResBlock(2 * hidden_channel, hidden_channel),
            nn.Conv3d(hidden_channel, output_channel, kernel_size=1, stride=1),
            nn.InstanceNorm3d(output_channel)
        )

    def forward(self, input_layer):
        tmp = input_layer.shape
        Vh, s, U = torch.linalg.svd(torch.reshape(input_layer, (tmp[0], tmp[1], tmp[2] * tmp[3], tmp[4])).permute(0, 1, 3, 2), full_matrices=False)
        Vh = Vh @ torch.sqrt(torch.diag_embed(s))
        U = (torch.sqrt(torch.diag_embed(s)) @ U).permute(0, 1, 3, 2)
        U = torch.reshape(U, (U.shape[0], U.shape[1], tmp[2], tmp[3], U.shape[3]))
        cat1, down1 = self.down1(U)
        cat2, down2 = self.down2(down1)
        cat3, down3 = self.down3(down2)
        bottom = self.bottom(down3)
        up3 = self.up3(torch.cat([cat3, bottom], dim=1))
        up2 = self.up2(torch.cat([cat2, up3], dim=1))
        output_U = self.outputs(torch.cat([cat1, up2], dim=1))
        output_U = torch.reshape(output_U, (output_U.shape[0], output_U.shape[1], output_U.shape[2] * output_U.shape[3],
                                             output_U.shape[4])).permute(0, 1, 3, 2)
        output_Vh = self.blockV(Vh) + Vh
        output_layer = (output_Vh @ output_U).permute(0, 1, 3, 2)
        output_layer = torch.reshape(output_layer, (output_layer.shape[0], output_layer.shape[1], tmp[2], tmp[3], output_layer.shape[3]))
        output_layer = nn.InstanceNorm3d(1)(output_layer)
        return output_layer
