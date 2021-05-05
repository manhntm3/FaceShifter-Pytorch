import torch
import torch.nn as nn

class AADLayer(nn.Module):
    def __init__(self, h_channel, z_channel, z_id_size=256):
        super(AADLayer, self).__init__()
        self.normalization = nn.BatchNorm2d(h_channel)
        self.conv_h = nn.Conv2d(h_channel, h_channel, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(z_channel, h_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(z_channel, h_channel, kernel_size=3, stride=1, padding=1, bias=True)

        self.fc_1 = nn.Linear(z_id_size, h_channel)
        self.fc_2 = nn.Linear(z_id_size, h_channel)


    def forward(self, h, z_att, z_id):
        h = self.normalization(h)
        M = self.sigmoid(self.conv_h(h))

        gamma_att = self.conv1(z_att)
        beta_att = self.conv2(z_att)
        A = gamma_att * h + beta_att

        gamma_id = self.fc_1(z_id)
        beta_id = self.fc_2(z_id)
        
        gamma_id = gamma_id.unsqueeze(-1).unsqueeze(-1).expand_as(h)
        beta_id = beta_id.unsqueeze(-1).unsqueeze(-1).expand_as(h)
        I = gamma_id * h + beta_id

        h_out = (1 - M) * A + M * I
        
        return h_out


class AADResBlock(nn.Module):
    def __init__(self, h_cin, z_channel, h_cout, z_id_size=256):
        super(AADResBlock, self).__init__()
        self.h_diff = h_cin != h_cout
        # First AAD Block h_cin -> h_cin
        self.AADLayer_1 = AADLayer(h_cin, z_channel, z_id_size)
        self.conv1 = nn.Conv2d(h_cin, h_cin, kernel_size=3, stride=1, padding=1)
        # Second AAD Block h_cin -> h_cout
        self.AADLayer_2 = AADLayer(h_cin, z_channel, z_id_size)
        self.conv2 = nn.Conv2d(h_cin, h_cout, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU()

        if self.h_diff:
            self.AADLayer_3 = AADLayer(h_cin, z_channel, z_id_size)
            self.conv3 = nn.Conv2d(h_cin, h_cout, kernel_size=3, stride=1, padding=1)

    def forward(self, h_in, z_att, z_id):
        x_res = h_in

        x = self.AADLayer_1(h_in, z_att, z_id)
        x = self.activation(x)
        x = self.conv1(x)

        x = self.AADLayer_2(x, z_att, z_id)
        x = self.activation(x)
        x = self.conv2(x)

        if self.h_diff:
            x_res = self.AADLayer_3(x_res, z_att, z_id)
            x_res = self.activation(x_res)
            x_res = self.conv3(x_res)
            
        return x + x_res


class AADGenerator(nn.Module):
    def __init__(self, z_id_size):
        super(AADGenerator, self).__init__()

        self.convt = nn.ConvTranspose2d(z_id_size, 1024, kernel_size=2, stride=1, padding=0)
        self.UpSample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.h_cin = [1024, 1024, 1024, 1024, 512, 256, 128, 64]
        self.h_cout = [1024, 1024, 1024, 512, 256, 128, 64, 3]
        self.z_channel = [1024, 2048, 1024, 512, 256, 128, 64, 64]

        self.model = nn.ModuleDict(
            {f'g_layer_{i}' : AADResBlock(self.h_cin[i], self.z_channel[i], self.h_cout[i], z_id_size)
            for i in range(8)}
        )
        
    def forward(self, z_id, z_atts):
        x = self.convt(z_id.unsqueeze(-1).unsqueeze(-1))
        for i in range(7):
            x = self.model[f'g_layer_{i}'](x, z_atts[i], z_id)
            x = self.UpSample(x)
        x = self.model[f'g_layer_7'](x, z_atts[7], z_id)
        return nn.Tanh()(x)
        # return nn.Sigmoid()(x)

class MultiLevelAttributeEncoder(nn.Module):
    def __init__(self):
        super(MultiLevelAttributeEncoder, self).__init__()
        self.encoder_channel = [3, 32, 64, 128, 256, 512, 1024, 1024] # Seven feature encoder extracted [32x128x128]->[1024x2x2]
        self.decoder_inchannel = [1024, 2048, 1024, 512, 256, 128] # Six feature decoder extracted
        self.decoder_outchannel = [1024, 512, 256, 128, 64, 32]
        self.encoder_size = len(self.encoder_channel)-1
        self.decoder_size = self.encoder_size-1
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
                nn.BatchNorm2d(self.decoder_outchannel[i]),
                nn.LeakyReLU(0.1)
            ) for i in range(self.decoder_size)
            }
        )
        self.UpSample = nn.UpsamplingBilinear2d(scale_factor=2)
    def forward(self, x):
        encoder_feature = []
        for i in range(self.encoder_size):
            x = self.Encoder[f'e_layer_{i}'](x)
            encoder_feature.append(x)
        decoder_feature = [encoder_feature[self.encoder_size-1]]
        y = encoder_feature[self.encoder_size-1]
        for i in range(self.decoder_size):
            y = self.Decoder[f'd_layer_{i}'](y)
            y = torch.cat((y, encoder_feature[self.decoder_size-1-i]), 1)
            decoder_feature.append(y)
        decoder_feature.append(self.UpSample(y))

        return decoder_feature