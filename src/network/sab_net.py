import torch
from torch import nn

from .nn import ConvDropoutNormReLU, SABlock, ABLayer


class SaBNet(nn.Module):
    def __init__(self, in_chs=1, out_chs=2, num_heads=2):
        super().__init__()

        self.encoders = nn.ModuleList([
            nn.Sequential(ConvDropoutNormReLU(in_chs, 32, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(32, 32, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(ConvDropoutNormReLU(32, 64, (3, 3, 3), (2, 2, 2)),
                          ConvDropoutNormReLU(64, 64, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(ConvDropoutNormReLU(64, 128, (3, 3, 3), (2, 2, 2)),
                          ConvDropoutNormReLU(128, 128, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(ConvDropoutNormReLU(128, 256, (3, 3, 3), (2, 2, 2)),
                          ConvDropoutNormReLU(256, 256, (3, 3, 3), (1, 1, 1)),
                          SABlock(3, 256, num_heads=num_heads, save_attn=True)),
            nn.Sequential(ConvDropoutNormReLU(256, 320, (3, 3, 3), (2, 2, 2)),
                          ConvDropoutNormReLU(320, 320, (3, 3, 3), (1, 1, 1)),
                          SABlock(3, 320, num_heads=num_heads)),
            nn.Sequential(ConvDropoutNormReLU(320, 320, (3, 3, 3), (1, 2, 2)),
                          ConvDropoutNormReLU(320, 320, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(ConvDropoutNormReLU(320, 320, (3, 3, 3), (1, 2, 2)),
                          ConvDropoutNormReLU(320, 320, (3, 3, 3), (1, 1, 1)))
        ])
        self.ab_layer = ABLayer(num_heads, (32, 64, 128, 256), (3, 3, 3), (1, 1, 1))

        self.stages = nn.ModuleList([
            nn.Sequential(ConvDropoutNormReLU(640, 320, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(320, 320, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(ConvDropoutNormReLU(640, 320, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(320, 320, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(ConvDropoutNormReLU(512, 256, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(256, 256, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(ConvDropoutNormReLU(256, 128, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(128, 128, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(ConvDropoutNormReLU(128, 64, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(64, 64, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(ConvDropoutNormReLU(64, 32, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(32, 32, (3, 3, 3), (1, 1, 1))),
        ])

        self.trans_convs = nn.ModuleList([
            nn.ConvTranspose3d(320, 320, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.ConvTranspose3d(320, 320, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.ConvTranspose3d(320, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        ])

        self.seg_layers = nn.ModuleList([
            nn.Conv3d(320, out_chs, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(320, out_chs, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(256, out_chs, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(128, out_chs, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(64, out_chs, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(32, out_chs, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        ])

        self.deep_supervision = False
        print(f'SaBNet initialized: in_chs={in_chs}, out_chs={out_chs}, num_heads={num_heads}')

    def train(self, mode: bool = True):
        super().train(mode)
        self.deep_supervision = True

    def eval(self):
        super().eval()
        self.deep_supervision = False

    def forward(self, x, epoch=None):
        skips = []
        t = x
        for encoder in self.encoders:
            t = encoder(t)
            skips.append(t)

        if not self.decoder.deep_supervision or (epoch is not None and epoch > 59):
            attn = self.encoders[3][2].attn
            skips = self.ab_layer(x, skips, attn)

        seg_outputs = []
        lup_inp = skips[-1]
        for i in range(len(self.stages)):
            x = self.trans_convs[i](lup_inp)
            x = torch.cat((x, skips[-(i+2)]), 1)
            x = self.stages[i](x)
            seg_outputs.append(self.seg_layers[i](x))
            lup_inp = x
        seg_outputs = seg_outputs[::-1]

        return seg_outputs if self.deep_supervision else seg_outputs[0]
