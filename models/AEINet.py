import torch
from torch import nn
from AADLayer import AADResBlock

class MultiLevelAttributeEncoder(nn.Module):
    def __init__(self):
        super(MultiLevelAttributeEncoder, self).__init__()
        self.encoder_channel = [3, 32, 64, 128, 256, 512, 1024, 1024]
        self.decoder_inchannel = [1024, 2048, 1024, 512, 256, 128]
        self.decoder_outchannel = [1024, 512, 256, 128, 64, 32]
        self.encoder_size = len(self.encoder_channel)-1
        self.Encoder = nn.ModuleDict(
            {f'e_layer_{i}' : nn.Sequential(
                nn.Conv2d(self.encoder_channel[i], self.encoder_channel[i+1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(self.encoder_channel[i+1]),
                nn.LeakyReLU(0.1)
            ) for i in range(self.encoder_size)
            }
        )
        self.Decoder = nn.ModuleDict(
            {f'd_layer_{i}' : nn.Sequential(
                nn.ConvTranspose2d(self.decoder_inchannel[i], self.decoder_outchannel[i], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(self.decoder_outchannel[i+1]),
                nn.LeakyReLU(0.1)
            ) for i in range(6)
            }
        )
        self.UpSample = nn.UpsamplingBilinear2d(scale_factor=2)
    def forward(self, x):
        encoder_feature = []
        for i in range(self.encoder_size):
            x = self.Encoder[f'e_layer_{i}'](x)
            encoder_feature.append(x)

        decoder_feature = [encoder_feature[6]]
        y = encoder_feature[6]
        for i in range(6):
            y = self.Decoder(f'd_layer_{i}')(y)
            y = torch.cat((y, encoder_feature[5-i]), 1)
            decoder_feature.append(y)
        decoder_feature.append(self.UpSample(y))

        return decoder_feature

class AADGenerator(nn.Module):
    def __init__(self, z_id_size):
        super(AADGenerator, self).__init__()

        self.convt = nn.ConvTranspose2d(z_id_size, 1024, kernel_size=2, stride=1, padding=0)
        self.UpSample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.h_cin = [1024, 1024, 1024, 1024, 512, 256, 128, 64]
        self.z_channel = [1024, 2048, 1024, 512, 256, 128, 64, 64]
        self.h_cout = [1024, 1024, 1024, 1024, 512, 256, 128, 64, 3]

        self.model = nn.ModuleDict(
            {f'g_layer_{i}' : AADResBlock(self.h_cin[i], self.z_channel[i], self.h_cout[i])
            for i in range(8)}
        )
        
    def forward(self, z_id, z_att):
        x = self.convt(z_id.unsqueeze(-1).unsqueeze(-1))
        for i in range(7):
            x = self.UpSample(self.model[f'g_layer_{i}'](x, z_att[i], z_id))
        x = self.model[f'g_layer_7'](x, z_att[7], z_id)
        return nn.Sigmoid()(x)
