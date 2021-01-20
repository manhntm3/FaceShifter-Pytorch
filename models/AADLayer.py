import torch
import torch.nn as nn

class AADLayer(nn.Module):
    def __init__(self, h_channel, z_channel, z_id_size=256):
        super(AADLayer, self).__init__()
        self.normalization = nn.BatchNorm2d(h_channel)
        self.conv_h = nn.Conv2d(h_channel, h_channel, h_channel, kernel_size=3, stride=1, padding=1)
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
        gamma_id = self.fc_1(z_id)
        beta_id = self.fc_2(z_id)

        A = gamma_att * h + beta_att

        gamma_id = gamma_id.unsqueeze(-1).unsqueeze(-1).expand_as(h)
        beta_id = gamma_id.unsqueeze(-1).unsqueeze(-1).expand_as(h)
        I = gamma_id * h + beta_id

        h_out = (1 - M) * A + M * I
        
        return h_out


class AADResBlock(nn.Module):
    def __init__(self, h_cin, z_channel, h_cout, z_id_size=256):
        super(AADResBlock, self).__init__()
        self.h_cin = h_cin
        self.h_cout = h_cout

        self.AADLayer_1 = AADLayer(h_cin, z_channel, z_id_size)
        self.AADLayer_2 = AADLayer(h_cin, z_channel, z_id_size)

        self.conv1 = nn.Conv2d(h_cin, h_cin, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(h_cin, h_cin, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU()

        if self.h_cin != self.h_cout:
            self.AADLayer_3 = AADLayer(h_cin, z_channel, z_id_size)
            self.conv3 = nn.Conv2d(h_cin, h_cin, kernel_size=3, stride=1, padidng=1)

    def forward(self, h_in, z_att, z_id):
        x_res = h_in

        x = self.activation(self.AADLayer_1(h_in, z_att, z_id))
        x = self.conv1(x)
        x = self.activation(self.AADLayer_2(x, z_att, z_id))
        x = self.conv2(x)

        if self.h_cin != self.h_cout:
            x_res = self.activation(self.AADLayer_3(x_res, z_att, z_id))
            x_res = self.conv3(x_res)
            
        return x + x_res


