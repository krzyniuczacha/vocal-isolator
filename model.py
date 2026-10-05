import torch

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, stride=2)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.lrelu = torch.nn.LeakyReLU(0.2)
        

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, is_dropout=False):
        super(Decoder, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, padding=2, stride=2, output_padding=1)
        self.bn = torch.nn.BatchNorm2d(out_channels) if not is_last else torch.nn.Identity()
        self.relu = torch.nn.ReLU() if not is_last else torch.nn.Identity()
        self.dropout = torch.nn.Dropout(0.5) if is_dropout else torch.nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class UNet(torch.nn.Module):
    def __init__(self, first_layer_channels=16, depth=6, eps=1e-5):
        super(UNet, self).__init__()
        self.first_layer_channels = first_layer_channels
        self.depth = depth
        self.eps = eps
        self.multiplier = 2 ** depth
        self.channels = [1]
        for i in range(self.depth):
            self.channels.append(first_layer_channels * 2 ** i)
            
        self.bn = torch.nn.BatchNorm2d(1)
        self.encoders = torch.nn.ModuleList([Encoder(self.channels[i], self.channels[i+1]) for i in range(depth)])
        self.decoders = torch.nn.ModuleList([Decoder(512, 256, is_dropout=True), Decoder(512, 128, is_dropout=True), Decoder(256, 64, is_dropout=True), Decoder(128, 32), Decoder(64, 16), Decoder(32, 1, is_last=True)])
        

    def forward(self, x):
        # odcinamy bin nyquista
        length = x.shape[-1]
        x = x[:, :, :1024, :]

        remainder = length % self.multiplier
        pad_amount = (self.multiplier - remainder) % self.multiplier

        x = torch.nn.functional.pad(x, (0, pad_amount), mode="constant", value=0)
        x = torch.log10(x + self.eps)
        x = self.bn(x)

        encoder_output = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_output.append(x)
        
        for i in range(self.depth):
            if i > 0:
                x = torch.cat((x, encoder_output[self.depth - 1 - i]), dim=1)

            x = self.decoders[i](x)

        x = torch.sigmoid(x)
        #przycinamy czas i doklejanmy spowrotem bin nyquista
        x = x[:, :, :, :length]
        nyq_bin = x[:, :, -1:, :]
        x = torch.cat((x, nyq_bin), dim=2)

        return x
        



