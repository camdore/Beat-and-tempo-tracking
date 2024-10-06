import torch.nn as nn
import torch


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 0))
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 0))
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(1, 8))

        self.dropout = nn.Dropout(0.1)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size=(1, 3))

    def forward(self, x):

        y = self.conv1(x)
        y = self.elu(y)
        y = self.dropout(y)
        y = self.pool(y)

        y = self.conv2(y)
        y = self.elu(y)
        y = self.dropout(y)
        y = self.pool(y)

        y = self.conv3(y)
        y = self.elu(y)

        y = y.view(-1, y.shape[1], y.shape[2])

        return y


class TCNBlock(nn.Module):
    def __init__(self, in_channels=16, out_channels=16, dilatation=None, dropout=0.1):
        super().__init__()

        padding = (dilatation * (5 - 1)) // 2
        self.dil_conv = nn.Conv1d(in_channels, out_channels, kernel_size=5, dilation=dilatation, padding=padding)
        self.elu = nn.ELU()
        self.spatial_dropout = nn.Dropout(dropout)
        self.conv_1d = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # weight initialisation by default is sampled from uniform distribution for the Conv1d

    def forward(self, x):

        res = self.conv_1d(x)

        y = self.dil_conv(x)
        y = self.elu(y)
        y = self.spatial_dropout(y)

        y_next_layer = y

        return y_next_layer


class TCNModel(nn.Module):
    def __init__(self, dilatation=None, num_layers=11):
        super().__init__()

        self.feature_extractor = FeatureExtractor()

        self.dense = nn.Linear(16, 1)
        self.dropout = nn.Dropout1d(0.1)

        # weight initialisation by default is sampled from uniform distribution for the Conv1d

        self.network = nn.ModuleList()

        if dilatation is None:
            dilations = [2 ** i for i in range(0, num_layers)]  # on crée les différentes dilatations pour chaque layer

        for i in range(0, 11):
            self.network = self.network.append(TCNBlock(dilatation=dilations[i]))  # on crée le réseau avec les 11 layers

    def forward(self, x):

        y = self.feature_extractor(x)

        for block in self.network:
            y_beat = block(y)

        y_beat = self.dropout(y_beat)

        y_beat_transposed = y_beat.transpose(1, 2)

        y_beat = self.dense(y_beat_transposed)
        y_beat = y_beat.squeeze(-1)

        return y_beat


if __name__ == "__main__":

    tensor = torch.rand(size=(64, 1, 3000, 81))
    print('Input shape :', tensor.shape)
    model = TCNModel()
    output_beat = model(tensor)
    print('Output shape :', output_beat.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters :', total_params)
    print('List of layers and their number of parameters:')
    for name, p in model.named_parameters():
        print(name, p.numel())
