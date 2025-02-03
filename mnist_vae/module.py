import torch


class Image28x28Encoder(torch.nn.Module):
    def __init__(self, num_image_channels: int, bottleneck_dim: int):
        super(Image28x28Encoder, self).__init__()

        self.bottleneck_dim = bottleneck_dim

        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv1 = torch.nn.Conv2d(num_image_channels, 16, 3, padding_mode='replicate')
        self.conv2 = torch.nn.Conv2d(16, 8, 3, padding_mode='replicate')
        self.conv3 = torch.nn.Conv2d(8, self.bottleneck_dim, 7)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size = x.shape[0]

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        y = self.conv3(x).reshape(batch_size, self.bottleneck_dim)

        return y


class Image28x28Decoder(torch.nn.Module):
    image_size = 28

    def __init__(self, num_image_channels: int, bottleneck_dim: int):
        super(Image28x28Decoder, self).__init__()

        self.bottleneck_dim = bottleneck_dim
        self.num_image_channels = num_image_channels

        transitional_dim = 16

        self.fc = nn.Linear(20, 8 * self.image_size * self.image_size)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.conv1 = torch.nn.Conv2d(8, 16, 3, padding_mode='replicate')
        self.conv2 = torch.nn.Conv2d(16, 4, 3, padding_mode='replicate')
        self.conv3 = torch.nn.Conv2d(4, num_image_channels, 3, padding_mode='replicate')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size = x.shape[0]

        x = self.fc(x).reshape(batch_size, 8, self.image_size * self.image_size)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        y = self.sigmoid(self.conv3(x)).reshape(batch_size, self.num_image_channels, self.image_size, self.image_size)

        return y

class DigitClassifier(torch.nn.Module):
    def __init__(self, bottleneck_dim: int):
        super(DigitClassifier, self).__init__()

        transitional_dim = 50

        self.module = torch.nn.Sequential(
            [
                torch.nn.Linear(bottleneck_dim, transitional_dim)
                torch.nn.ReLU(),
                torch.nn.Linear(transitional_dim, 10),
            ]
        )

    def forward(x: torch.Tensor) -> torch.Tensor:

        y = self.module(x)

        n = torch.argmax(y)

        return n


class VAE(torch.nn.Module):
    def __init__(self, bottleneck_dim: int, num_image_channels: int):
        super(VAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.mean_layer = torch.nn.Linear(, bottleneck_dim)
