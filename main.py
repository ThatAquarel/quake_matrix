import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectogramCNN(nn.Module):
    def __init__(self,num_classes):
        super(SpectogramCNN, self).__init__()


        self.first_conv_layer = nn.Conv2d(in_channels=1,out_channels=20, kernel_size=4, stride=1, padding=1)

        self.pool_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.second_conv_layer = nn.Conv2d(20,40, kernel_size=4, stride=1, padding=1)

        self.third_conv_layer = nn.Conv2d(40 , 80, kernel_size=4, stride=1, padding=1)

        self.first_connect_layer = nn.Linear(80 * 30 * 30, out_features=256)

        self.second_connect_layer = nn.Linear(256, num_classes)


    def forward(self, x):
        x = self.pool_layer(F.relu(self.first_conv_layer(x)))
        x = self.pool_layer(F.relu(self.second_conv_layer(x)))
        x = self.pool_layer(F.relu(self.third_conv_layer(x)))

        x = x.view(-1, 80 * 30 * 30)
        x = F.relu(self.first_connect_layer(x))
        x = self.second_connect_layer(x)

        return x


num_classes = 2
model = SpectogramCNN(num_classes)


test_input = torch.randn(1, 1, 256, 256)

output = model(test_input)

print(output.shape)
